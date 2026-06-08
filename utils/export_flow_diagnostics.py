"""
导出新旧模型的整图 flow / error / folding 诊断面板。

这个脚本面向当前水下裂缝扭曲校正项目的下一步排查：
1. 固定一批由 `compare_eval_per_image.py` 挑出的典型样本；
2. 同时加载旧模型和 v2 新模型；
3. 对每张图导出输入、旧模型校正、新模型校正、GT 校正、EPE 热图、
   folding 热区和预测位移幅度图；
4. 额外输出逐图 CSV，便于把可视化现象和指标退化对应起来。

注意：这里的 flow 仍沿用项目既有约定，即网络输出为归一化逆映射坐标场，
`flow[:, 0]` 和 `flow[:, 1]` 分别对应 `grid_sample` 的 x/y 采样坐标。
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from loss_crack import CrackMaskEstimator
from models.crack_warp_net import build_crack_warp_net


def load_model(model_path: str, device: torch.device):
    """加载单个 checkpoint，并兼容纯 state_dict 或带 state_dict/model 键的保存格式。"""
    net = build_crack_warp_net().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
    cleaned = {}
    for key, value in ckpt.items():
        cleaned[key.replace("module.", "")] = value
    net.load_state_dict(cleaned, strict=False)
    net.eval()
    return net


def load_image_paths(img_dir: str, image_list: str, num: int):
    """
    加载需要诊断的图片路径。

    如果提供 `image_list`，每行既可以是图片文件名，也可以是绝对路径；
    否则从 `img_dir` 中按文件名排序取前 num 张。
    """
    if image_list:
        out = []
        with open(image_list, "r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                path = item if os.path.isabs(item) else os.path.join(img_dir, item)
                if os.path.exists(path):
                    out.append(path)
                else:
                    print(f"[WARN] image not found, skipped: {path}")
    else:
        out = []
        for name in sorted(os.listdir(img_dir)):
            low = name.lower()
            if low.endswith((".png", ".jpg", ".jpeg")) and not low.endswith(".npy"):
                out.append(os.path.join(img_dir, name))

    if num > 0:
        out = out[:num]
    return out


def image_to_tensor(img_bgr: np.ndarray, size: int):
    """将 OpenCV BGR 图片转为模型输入 tensor，同时返回 resize 后的 BGR 图片。"""
    img_512 = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = img_512[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_512, tensor


def load_label(label_path: str, size: int):
    """读取归一化逆映射标签，并 resize 到诊断尺寸。"""
    label = np.load(label_path).astype(np.float32)
    if label.ndim == 3 and label.shape[0] == 2:
        label = label.transpose(1, 2, 0)
    if label.shape[:2] != (size, size):
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(label.transpose(2, 0, 1)).float().unsqueeze(0)


def flow_to_grid(flow: torch.Tensor):
    """把归一化 [0,1] 逆映射坐标转为 grid_sample 需要的 [-1,1] grid。"""
    gx = flow[:, 0] * 2.0 - 1.0
    gy = flow[:, 1] * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1)


def run_model(net, img_t: torch.Tensor):
    """执行模型推理，并统一处理迭代网络返回 list 的情况。"""
    with torch.no_grad():
        flow = net(img_t)
        if isinstance(flow, list):
            flow = flow[-1]
    return flow.clamp(0.0, 1.0)


def warp_image(img_t: torch.Tensor, flow: torch.Tensor):
    """按逆映射 flow 对输入图像做校正。"""
    warped = F.grid_sample(img_t, flow_to_grid(flow), mode="bilinear", padding_mode="border", align_corners=True)
    arr = warped[0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def identity_flow(size: int, device: torch.device):
    """生成归一化恒等坐标场，用于计算预测恢复幅度。"""
    ys, xs = torch.meshgrid(
        torch.linspace(0.0, 1.0, size, device=device),
        torch.linspace(0.0, 1.0, size, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=0).unsqueeze(0)


def epe_map_px(pred: torch.Tensor, gt: torch.Tensor, size: int):
    """计算逐像素 EPE，单位为像素。"""
    return torch.sqrt(((pred - gt) ** 2).sum(dim=1).clamp_min(1e-12)) * float(size - 1)


def folding_heat(flow: torch.Tensor):
    """
    计算 folding 热区。

    与当前评估脚本保持一致：直接在归一化采样坐标上用中心差分估计 Jacobian，
    determinant 小于 0 的区域视为局部折叠。返回值 shape 为 (H, W)。
    """
    u = flow[:, 0]
    v = flow[:, 1]
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) * 0.5
    du_dy = (u[:, 2:, :] - u[:, :-2, :]) * 0.5
    dv_dx = (v[:, :, 2:] - v[:, :, :-2]) * 0.5
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) * 0.5
    det = du_dx[:, 1:-1, :] * dv_dy[:, :, 1:-1] - du_dy[:, :, 1:-1] * dv_dx[:, 1:-1, :]
    fold = (det < 0).float()
    # 中心差分会少掉一圈边界，这里补回到原图大小，方便直接可视化叠加。
    return F.pad(fold, (1, 1, 1, 1), mode="constant", value=0.0)


def to_heatmap(values: np.ndarray, vmax=None, cmap=cv2.COLORMAP_TURBO):
    """把单通道数值图转换成伪彩色热图。"""
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if vmax is None:
        vmax = float(np.percentile(values, 95))
    vmax = max(vmax, 1e-6)
    norm = np.clip(values / vmax, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap)


def add_title(img: np.ndarray, title: str):
    """给面板子图加黑色标题栏，避免文字压到图像细节上。"""
    h, w = img.shape[:2]
    bar = np.zeros((28, w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 255, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def resize_for_panel(img: np.ndarray, width: int):
    """统一子图宽度，防止不同图尺寸导致拼接错位。"""
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def make_panel(stem, inp, old_warp, new_warp, gt_warp, old_epe, new_epe, old_fold, new_fold, old_mag, new_mag):
    """生成整图诊断面板，按两行展示校正结果、误差、folding 和恢复幅度。"""
    tile_w = 320
    epe_vmax = max(float(np.percentile(old_epe, 95)), float(np.percentile(new_epe, 95)), 1.0)
    mag_vmax = max(float(np.percentile(old_mag, 95)), float(np.percentile(new_mag, 95)), 1.0)

    tiles = [
        add_title(resize_for_panel(inp, tile_w), f"{stem} | input"),
        add_title(resize_for_panel(old_warp, tile_w), "old corrected"),
        add_title(resize_for_panel(new_warp, tile_w), "v2 corrected"),
        add_title(resize_for_panel(gt_warp, tile_w), "gt corrected"),
        add_title(resize_for_panel(to_heatmap(old_epe, vmax=epe_vmax), tile_w), "old EPE heat"),
        add_title(resize_for_panel(to_heatmap(new_epe, vmax=epe_vmax), tile_w), "v2 EPE heat"),
        add_title(resize_for_panel(to_heatmap(old_fold, vmax=1.0, cmap=cv2.COLORMAP_HOT), tile_w), "old folding"),
        add_title(resize_for_panel(to_heatmap(new_fold, vmax=1.0, cmap=cv2.COLORMAP_HOT), tile_w), "v2 folding"),
        add_title(resize_for_panel(to_heatmap(old_mag, vmax=mag_vmax), tile_w), "old disp mag"),
        add_title(resize_for_panel(to_heatmap(new_mag, vmax=mag_vmax), tile_w), "v2 disp mag"),
    ]

    row1 = np.hstack(tiles[:5])
    row2 = np.hstack(tiles[5:])
    return np.vstack([row1, row2])


def scalar_mean(tensor: torch.Tensor):
    """安全取 tensor 均值为 Python float。"""
    return float(tensor.detach().mean().cpu().item())


def main():
    parser = argparse.ArgumentParser(description="Export old/new full-image flow diagnostics")
    parser.add_argument("--old_model", required=True, help="旧模型 checkpoint")
    parser.add_argument("--new_model", required=True, help="新模型 checkpoint")
    parser.add_argument("--img_dir", required=True, help="图片和 .npy 标签目录")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--image_list", default=None, help="可选图片清单，每行一个文件名或路径")
    parser.add_argument("--num", type=int, default=-1, help="最多处理多少张，-1 表示全部")
    parser.add_argument("--size", type=int, default=512, help="诊断尺寸")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--crack_topk", type=float, default=0.08, help="裂缝 soft mask topk")
    parser.add_argument("--crack_temp", type=float, default=0.07, help="裂缝 soft mask temperature")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    old_net = load_model(args.old_model, device)
    new_net = load_model(args.new_model, device)
    masker = CrackMaskEstimator(topk=args.crack_topk, temperature=args.crack_temp).to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = load_image_paths(args.img_dir, args.image_list, args.num)
    if not paths:
        raise RuntimeError("No images selected for diagnostics")

    rows = []
    for idx, img_path in enumerate(paths, start=1):
        label_path = img_path + ".npy"
        if not os.path.exists(label_path):
            print(f"[WARN] label not found, skipped: {label_path}")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] failed to read image, skipped: {img_path}")
            continue

        inp, img_t_cpu = image_to_tensor(img_bgr, args.size)
        img_t = img_t_cpu.to(device)
        gt = load_label(label_path, args.size).to(device)

        old_flow = run_model(old_net, img_t)
        new_flow = run_model(new_net, img_t)
        ident = identity_flow(args.size, device)
        with torch.no_grad():
            crack_mask = masker(img_t)

        old_epe_t = epe_map_px(old_flow, gt, args.size)
        new_epe_t = epe_map_px(new_flow, gt, args.size)
        old_fold_t = folding_heat(old_flow)
        new_fold_t = folding_heat(new_flow)
        old_mag_t = torch.sqrt(((old_flow - ident) ** 2).sum(dim=1).clamp_min(1e-12)) * float(args.size - 1)
        new_mag_t = torch.sqrt(((new_flow - ident) ** 2).sum(dim=1).clamp_min(1e-12)) * float(args.size - 1)

        # 裂缝区域指标只用于诊断排序，mask 非人工标注，只代表当前项目一致的 soft crack attention。
        mask_den = crack_mask.sum().clamp_min(1.0)
        old_crack_epe = float((old_epe_t.unsqueeze(1) * crack_mask).sum().detach().cpu().item() / mask_den.detach().cpu().item())
        new_crack_epe = float((new_epe_t.unsqueeze(1) * crack_mask).sum().detach().cpu().item() / mask_den.detach().cpu().item())

        old_warp = warp_image(img_t, old_flow)
        new_warp = warp_image(img_t, new_flow)
        gt_warp = warp_image(img_t, gt)

        old_epe = old_epe_t[0].detach().cpu().numpy()
        new_epe = new_epe_t[0].detach().cpu().numpy()
        old_fold = old_fold_t[0].detach().cpu().numpy()
        new_fold = new_fold_t[0].detach().cpu().numpy()
        old_mag = old_mag_t[0].detach().cpu().numpy()
        new_mag = new_mag_t[0].detach().cpu().numpy()

        stem = Path(img_path).stem
        panel = make_panel(stem, inp, old_warp, new_warp, gt_warp, old_epe, new_epe, old_fold, new_fold, old_mag, new_mag)
        cv2.imwrite(str(out_dir / f"{stem}_flow_diagnostic.png"), panel)

        rows.append({
            "image": Path(img_path).name,
            "old_global_epe_px": scalar_mean(old_epe_t),
            "new_global_epe_px": scalar_mean(new_epe_t),
            "old_crack_epe_px": old_crack_epe,
            "new_crack_epe_px": new_crack_epe,
            "old_folding_rate": scalar_mean(old_fold_t),
            "new_folding_rate": scalar_mean(new_fold_t),
            "old_disp_mag_mean_px": scalar_mean(old_mag_t),
            "new_disp_mag_mean_px": scalar_mean(new_mag_t),
            "old_disp_mag_p95_px": float(np.percentile(old_mag, 95)),
            "new_disp_mag_p95_px": float(np.percentile(new_mag, 95)),
        })
        print(f"[{idx}/{len(paths)}] {stem}: old_epe={rows[-1]['old_global_epe_px']:.3f}, new_epe={rows[-1]['new_global_epe_px']:.3f}")

    csv_path = out_dir / "flow_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Images processed: {len(rows)}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
