#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 UniMatch 外部对比方法的预测坐标场。

背景：
    本项目主模型是单图像预测逆映射坐标场；UniMatch 是双图像 dense matching /
    optical flow 方法。为了把 UniMatch 纳入对比实验，需要先把它的输出统一转换成
    本项目评估脚本需要的 `pred_grid/xxx.png.npy`。

输入与方向：
    - 输入图：`underwater_crack_v3/xxx.png`，即水下扭曲裂缝图。
    - GT 标签：`underwater_crack_v3/xxx.png.npy`，即校正图像像素到输入图像像素的
      归一化逆映射坐标场。
    - 本脚本先用 GT 标签生成一张 GT 校正图，然后让 UniMatch 估计：
          GT 校正图 -> 原始扭曲输入图
      的像素位移。
    - 这个方向和项目标签一致：输出校正图上的每个像素应采样输入图的哪个位置。

重要说明：
    这是一种“oracle pair”密集匹配对比：UniMatch 使用了 GT 校正图作为匹配目标。
    它不能代表单图像校正模型的同等输入条件，但可以作为近年 dense matching 方法的
    强相关参考基线。报告中需要明确这一点。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_metrics import flow_to_grid, load_image_tensor, load_label_tensor


def list_image_label_pairs(img_dir: Path) -> List[Tuple[Path, Path]]:
    """列出有 `.npy` 标签的图像样本，保持和评估脚本一致的顺序。"""

    rows: List[Tuple[Path, Path]] = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = img_dir / name
        label_path = Path(str(img_path) + ".npy")
        if label_path.exists():
            rows.append((img_path, label_path))
    return rows


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 32) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """把图像 padding 到指定倍数，避免 UniMatch 下采样时报尺寸问题。

    返回：
        padded tensor，以及原始 `(height, width)`，后续用于裁剪 flow。
    """

    _, _, height, width = tensor.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return padded, (height, width)


def unpad_flow(flow: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """把 UniMatch 输出 flow 裁回原始尺寸。"""

    height, width = size
    return flow[:, :, :height, :width]


def image_to_unimatch_input(img: torch.Tensor) -> torch.Tensor:
    """把 `[0,1]` RGB tensor 转成 UniMatch 常用的 `[0,255]` 输入。"""

    return img * 255.0


def make_gt_corrected(input_img: torch.Tensor, gt_grid: torch.Tensor) -> torch.Tensor:
    """使用 GT 逆映射坐标场生成校正图。

    参数：
        input_img: `(1,3,H,W)`，扭曲输入图。
        gt_grid: `(1,2,H,W)`，归一化逆映射坐标场。
    """

    return F.grid_sample(
        input_img,
        flow_to_grid(gt_grid),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def load_unimatch(repo_dir: Path, checkpoint: Path, device: torch.device):
    """加载 UniMatch 模型并恢复 checkpoint。"""

    sys.path.insert(0, str(repo_dir))
    from unimatch.unimatch import UniMatch  # pylint: disable=import-error,import-outside-toplevel

    # 参数对应官方 GMFlow scale2 reg-refine6 mixed-data checkpoint。
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow",
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state: Dict[str, torch.Tensor]
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def pixel_flow_to_normalized_grid(flow: torch.Tensor) -> np.ndarray:
    """将像素位移 flow 转为 `[0,1]` 归一化逆映射坐标场。

    UniMatch 估计的是从 GT 校正图到扭曲输入图的像素位移，所以 identity + flow
    就是本项目所需的逆映射坐标。
    """

    _, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    gx = (x + flow[0, 0]).clamp(0, width - 1) / float(width - 1)
    gy = (y + flow[0, 1]).clamp(0, height - 1) / float(height - 1)
    grid = torch.stack([gx, gy], dim=0)
    return grid.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def export_predictions(args: argparse.Namespace) -> None:
    """运行 UniMatch 并导出 `pred_grid/*.npy`。"""

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_unimatch(
        repo_dir=Path(args.unimatch_repo),
        checkpoint=Path(args.checkpoint),
        device=device,
    )

    rows = list_image_label_pairs(img_dir)
    if args.num > 0:
        rows = rows[: args.num]
    if not rows:
        raise RuntimeError(f"没有找到图像和标签配对: {img_dir}")

    for index, (img_path, label_path) in enumerate(rows, start=1):
        output_path = out_dir / f"{img_path.name}.npy"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(rows)}] skip existing: {output_path}")
            continue

        input_img = load_image_tensor(str(img_path), size=(args.size, args.size)).unsqueeze(0).to(device)
        gt_grid = load_label_tensor(str(label_path), size=(args.size, args.size)).unsqueeze(0).to(device)
        corrected_img = make_gt_corrected(input_img, gt_grid)

        image1 = image_to_unimatch_input(corrected_img)
        image2 = image_to_unimatch_input(input_img)
        image1, original_size = pad_to_multiple(image1, multiple=32)
        image2, _ = pad_to_multiple(image2, multiple=32)

        result = model(
            image1,
            image2,
            attn_type=args.attn_type,
            attn_splits_list=args.attn_splits_list,
            corr_radius_list=args.corr_radius_list,
            prop_radius_list=args.prop_radius_list,
            num_reg_refine=args.num_reg_refine,
            task="flow",
        )
        flow = result["flow_preds"][-1]
        flow = unpad_flow(flow, original_size)
        pred_grid = pixel_flow_to_normalized_grid(flow)
        np.save(output_path, pred_grid)

        if index % args.log_interval == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] saved: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Export UniMatch predictions as normalized inverse grids")
    parser.add_argument("--img_dir", default="underwater_crack_v3", help="数据目录")
    parser.add_argument(
        "--out_dir",
        default="output_crackwarp_slurm/external_baselines/unimatch/pred_grid",
        help="预测坐标场输出目录",
    )
    parser.add_argument(
        "--unimatch_repo",
        default="external_methods/unimatch",
        help="UniMatch 仓库目录，建议放在项目内 external_methods/unimatch。",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "external_methods/unimatch/pretrained/"
            "gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
        ),
        help="UniMatch flow 预训练权重。",
    )
    parser.add_argument("--num", type=int, default=10, help="导出样本数，-1 表示全量")
    parser.add_argument("--size", type=int, default=512, help="图像尺寸")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在预测文件")
    parser.add_argument("--log-interval", type=int, default=10, help="日志打印间隔")

    # 以下参数和官方 demo 常用 GMFlow 配置保持一致。
    parser.add_argument("--attn-type", default="swin", help="UniMatch attention 类型")
    parser.add_argument("--attn-splits-list", type=int, nargs="+", default=[2, 8], help="多尺度 attention split")
    parser.add_argument("--corr-radius-list", type=int, nargs="+", default=[-1, 4], help="多尺度 correlation radius")
    parser.add_argument("--prop-radius-list", type=int, nargs="+", default=[-1, 1], help="多尺度 propagation radius")
    parser.add_argument("--num-reg-refine", type=int, default=6, help="refinement 迭代次数")
    return parser.parse_args()


if __name__ == "__main__":
    export_predictions(parse_args())
