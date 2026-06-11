#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 SEA-RAFT 外部对比方法的预测坐标场。

SEA-RAFT 是 2024 年光流方法，适合作为新近 dense matching / optical flow
对比。和 `export_unimatch_predictions.py` 类似，本脚本采用 oracle-pair 设置：

    GT 校正图 -> 原始扭曲输入图

并将 SEA-RAFT 输出的像素 flow 转换成项目统一的 `[0,1]` 归一化逆映射坐标场，
保存为 `pred_grid/xxx.png.npy`。报告中需要明确：该设置使用 GT 校正图作为匹配
参考，不是和单图像主模型完全相同的输入条件。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_metrics import flow_to_grid, load_image_tensor, load_label_tensor


def list_image_label_pairs(img_dir: Path) -> List[Tuple[Path, Path]]:
    """列出有 `.npy` 标签的图像样本，顺序与评估脚本保持一致。"""

    rows: List[Tuple[Path, Path]] = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = img_dir / name
        label_path = Path(str(img_path) + ".npy")
        if label_path.exists():
            rows.append((img_path, label_path))
    return rows


def make_gt_corrected(input_img: torch.Tensor, gt_grid: torch.Tensor) -> torch.Tensor:
    """用 GT 逆映射坐标场生成 GT 校正图。"""

    return F.grid_sample(
        input_img,
        flow_to_grid(gt_grid),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def image_to_searaft_input(img: torch.Tensor) -> torch.Tensor:
    """SEA-RAFT 内部会做归一化，这里输入 `[0,255]` RGB tensor。"""

    return img * 255.0


def load_config(cfg_path: Path) -> SimpleNamespace:
    """读取 SEA-RAFT JSON 配置为 Namespace。"""

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SimpleNamespace(**data)


def load_searaft(repo_dir: Path, cfg_path: Path, checkpoint: str, url: str, device: torch.device):
    """加载 SEA-RAFT 模型，支持本地 checkpoint 或 HuggingFace url。

    SEA-RAFT 仓库内部也有名为 `utils` 的包，而本项目同样有 `utils` 包。
    当前脚本前面已经导入了 `utils.evaluate_metrics`，如果不隔离导入环境，
    SEA-RAFT 的 `from utils.utils import coords_grid` 会误命中本项目的
    `utils/utils.py`，导致缺少 `coords_grid`。这里在导入 SEA-RAFT 前临时
    移除 `sys.modules` 里的项目 `utils` 条目，并把 SEA-RAFT 的 `core`
    目录放到搜索路径最前面，确保其内部依赖解析到自己的实现。
    """

    repo_dir = repo_dir.resolve()
    core_dir = repo_dir / "core"
    original_sys_path = list(sys.path)
    removed_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "utils" or name.startswith("utils.")
    }
    for name in removed_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(core_dir))
    sys.path.insert(0, str(repo_dir))

    try:
        raft_module = importlib.import_module("raft")
        sea_utils = importlib.import_module("utils.utils")
    finally:
        # 保持后续项目代码的导入环境干净；SEA-RAFT 已导入的模块对象会继续持有
        # 自己需要的函数引用。
        sys.path[:] = original_sys_path
        for name in list(sys.modules):
            if name == "utils" or name.startswith("utils."):
                sys.modules.pop(name, None)
        sys.modules.update(removed_modules)

    RAFT = raft_module.RAFT
    load_ckpt = sea_utils.load_ckpt

    args = load_config(cfg_path)
    args.cfg = str(cfg_path)
    args.path = checkpoint or None
    args.url = url or None
    args.device = "cuda" if device.type == "cuda" else "cpu"

    if args.path:
        model = RAFT(args)
        load_ckpt(model, args.path)
    elif args.url:
        model = RAFT.from_pretrained(args.url, args=args)
    else:
        raise ValueError("必须提供 --checkpoint 或 --url 用于加载 SEA-RAFT 权重。")

    model = model.to(device)
    model.eval()
    return model, args


def run_searaft(model, args: SimpleNamespace, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """按照 SEA-RAFT `custom.py` 的流程推理并还原到原图尺寸。"""

    scale = float(2 ** getattr(args, "scale", 0))
    image1_scaled = F.interpolate(image1, scale_factor=scale, mode="bilinear", align_corners=False)
    image2_scaled = F.interpolate(image2, scale_factor=scale, mode="bilinear", align_corners=False)

    output = model(image1_scaled, image2_scaled, iters=args.iters, test_mode=True)
    flow = output["flow"][-1]
    flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * (1.0 / scale)
    return flow


def pixel_flow_to_normalized_grid(flow: torch.Tensor) -> np.ndarray:
    """将 SEA-RAFT 像素 flow 转换为 `[0,1]` 归一化逆映射坐标场。"""

    _, _, height, width = flow.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    gx = (x + flow[0, 0]).clamp(0, width - 1) / float(width - 1)
    gy = (y + flow[0, 1]).clamp(0, height - 1) / float(height - 1)
    return torch.stack([gx, gy], dim=0).detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def export_predictions(args: argparse.Namespace) -> None:
    """运行 SEA-RAFT 并导出预测坐标场。"""

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = load_searaft(
        repo_dir=Path(args.searaft_repo),
        cfg_path=Path(args.cfg),
        checkpoint=args.checkpoint,
        url=args.url,
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

        image1 = image_to_searaft_input(corrected_img)
        image2 = image_to_searaft_input(input_img)
        flow = run_searaft(model, model_args, image1, image2)
        pred_grid = pixel_flow_to_normalized_grid(flow)
        np.save(output_path, pred_grid)

        if index % args.log_interval == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] saved: {output_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Export SEA-RAFT predictions as normalized inverse grids")
    parser.add_argument("--img_dir", default="underwater_crack_v3", help="数据目录")
    parser.add_argument(
        "--out_dir",
        default="output_crackwarp_slurm/external_baselines/searaft/pred_grid",
        help="预测坐标场输出目录",
    )
    parser.add_argument("--searaft_repo", default="external_methods/SEA-RAFT", help="SEA-RAFT 仓库目录")
    parser.add_argument("--cfg", default="external_methods/SEA-RAFT/config/eval/spring-M.json", help="SEA-RAFT 配置文件")
    parser.add_argument("--checkpoint", default="", help="本地 SEA-RAFT checkpoint；为空时使用 --url")
    parser.add_argument(
        "--url",
        default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
        help="HuggingFace 模型名；默认使用 README 推荐的 Spring-M 权重。",
    )
    parser.add_argument("--num", type=int, default=10, help="导出样本数，-1 表示全量")
    parser.add_argument("--size", type=int, default=512, help="图像尺寸")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在预测文件")
    parser.add_argument("--log-interval", type=int, default=10, help="日志打印间隔")
    return parser.parse_args()


if __name__ == "__main__":
    export_predictions(parse_args())
