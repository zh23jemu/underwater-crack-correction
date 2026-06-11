#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估外部对比方法输出的稠密坐标场。

用途：
    当前项目自己的 CrackWarpNet 可以直接用 `utils/evaluate_metrics.py` 评估；
    但 UniMatch、SEA-RAFT、MemFlow、NeuFlow 等外部方法通常不会保存成
    PyTorch checkpoint，也不一定能直接接入项目模型结构。为了公平比较，
    这些方法只需要先把每张图的预测结果导出为 `.npy` 坐标场，本脚本就能
    使用和主模型一致的 crack EPE、Dice、edge fidelity、folding rate 口径评估。

预测文件约定：
    1. 默认每张输入图 `xxx.png` 对应预测文件 `PRED_DIR/xxx.png.npy`。
    2. 支持 shape 为 `(2,H,W)` 或 `(H,W,2)`。
    3. 默认格式为 `normalized_grid`，即两个通道都是 `[0,1]` 范围的逆映射坐标，
       与本项目标签 `xxx.png.npy` 的格式一致。
    4. 如外部方法输出像素位移 `(dx,dy)`，可使用 `--pred-format pixel_flow`，
       脚本会将位移加到 identity grid 后转为归一化坐标场。

注意：
    光流方法通常预测“图 A 到图 B 的位移”，而本项目评估需要“扭曲图到校正图的
    逆映射坐标”。外部方法适配时必须先确认方向；如果方向反了，指标会很差。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss_crack import CrackMaskEstimator
from utils.evaluate_metrics import (  # 复用主模型评估口径，避免指标漂移。
    build_crack_mask_batch,
    edge_fidelity,
    flow_epe_px,
    flow_epe_px_masked,
    folding_rate,
    load_image_tensor,
    load_label_tensor,
    warp_crack_dice,
)


def list_image_label_pairs(img_dir: Path) -> List[Tuple[Path, Path]]:
    """列出存在 `.npy` 标签的图像样本。"""

    rows: List[Tuple[Path, Path]] = []
    for name in sorted(os.listdir(img_dir)):
        low = name.lower()
        if not low.endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = img_dir / name
        label_path = Path(str(img_path) + ".npy")
        if label_path.exists():
            rows.append((img_path, label_path))
    return rows


def identity_grid(size: int) -> np.ndarray:
    """生成 `[0,1]` 范围的 identity 采样网格，shape 为 `(2,H,W)`。"""

    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    return np.stack([gx, gy], axis=0)


def normalize_pred_array(pred: np.ndarray, size: int, pred_format: str) -> np.ndarray:
    """把外部方法预测结果转换为 `(2,size,size)` 的归一化坐标场。

    参数：
        pred: 外部方法保存的 numpy 数组。
        size: 评估分辨率，默认 512。
        pred_format: `normalized_grid` 或 `pixel_flow`。

    返回：
        与项目标签同格式的归一化逆映射坐标场，范围会被裁剪到 `[0,1]`。
    """

    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim != 3:
        raise ValueError(f"预测数组必须是 3 维，实际 shape={pred.shape}")

    # 兼容 `(H,W,2)` 和 `(2,H,W)` 两种常见保存方式。
    if pred.shape[-1] == 2:
        pred = pred.transpose(2, 0, 1)
    if pred.shape[0] != 2:
        raise ValueError(f"预测数组第一个维度应为 2，实际 shape={pred.shape}")

    if pred.shape[1:] != (size, size):
        pred = np.stack(
            [
                cv2.resize(pred[0], (size, size), interpolation=cv2.INTER_LINEAR),
                cv2.resize(pred[1], (size, size), interpolation=cv2.INTER_LINEAR),
            ],
            axis=0,
        ).astype(np.float32)

    if pred_format == "normalized_grid":
        grid = pred
    elif pred_format == "pixel_flow":
        # 外部光流常用像素位移，单位是 pixel。这里把它转换成归一化采样坐标。
        # 正负方向必须由适配脚本保证；本函数只负责格式转换。
        grid = identity_grid(size)
        grid[0] = grid[0] + pred[0] / float(size - 1)
        grid[1] = grid[1] + pred[1] / float(size - 1)
    else:
        raise ValueError(f"未知预测格式: {pred_format}")

    return np.clip(grid, 0.0, 1.0).astype(np.float32)


def resolve_prediction_path(pred_dir: Path, image_name: str, suffix: str) -> Path:
    """根据图像名推导预测文件路径。"""

    return pred_dir / f"{image_name}{suffix}"


def load_prediction_tensor(pred_path: Path, size: int, pred_format: str) -> torch.Tensor:
    """读取单个外部预测坐标场，并转换为 torch tensor。"""

    if not pred_path.exists():
        raise FileNotFoundError(f"缺少预测文件: {pred_path}")
    pred = np.load(pred_path)
    pred = normalize_pred_array(pred, size=size, pred_format=pred_format)
    return torch.from_numpy(pred).float()


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> None:
    """执行外部预测结果评估并写出 summary/per-image CSV。"""

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    img_dir = Path(args.img_dir)
    pred_dir = Path(args.pred_dir)
    rows = list_image_label_pairs(img_dir)
    if args.num > 0:
        rows = rows[: args.num]
    if not rows:
        raise RuntimeError(f"没有找到图像和标签配对: {img_dir}")

    crack_masker = CrackMaskEstimator(
        topk=args.crack_topk,
        temperature=args.crack_temp,
    ).to(device)
    crack_masker.eval()

    global_epe_all: List[np.ndarray] = []
    crack_epe_all: List[np.ndarray] = []
    global_edge_all: List[np.ndarray] = []
    crack_edge_all: List[np.ndarray] = []
    fold_all: List[np.ndarray] = []
    warp_dice_all: List[np.ndarray] = []
    crack_ratio_all: List[np.ndarray] = []
    per_image: List[Dict[str, float | str]] = []

    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i : i + args.batch_size]
        imgs = torch.stack(
            [load_image_tensor(str(img), size=(args.size, args.size)) for img, _ in chunk],
            dim=0,
        ).to(device)
        gts = torch.stack(
            [load_label_tensor(str(label), size=(args.size, args.size)) for _, label in chunk],
            dim=0,
        ).to(device)
        preds = torch.stack(
            [
                load_prediction_tensor(
                    resolve_prediction_path(pred_dir, img.name, args.pred_suffix),
                    size=args.size,
                    pred_format=args.pred_format,
                )
                for img, _ in chunk
            ],
            dim=0,
        ).to(device)

        crack_mask = build_crack_mask_batch(imgs, crack_masker)
        global_epe = flow_epe_px(preds, gts)
        crack_epe = flow_epe_px_masked(preds, gts, crack_mask)
        global_edge = edge_fidelity(preds, gts, imgs, mask=None)
        crack_edge = edge_fidelity(preds, gts, imgs, mask=crack_mask)
        fold = folding_rate(preds)
        warp_dice = warp_crack_dice(preds, gts, imgs, crack_masker=crack_masker)
        crack_ratio = crack_mask.mean(dim=(1, 2, 3))

        global_epe_all.append(global_epe.cpu().numpy())
        crack_epe_all.append(crack_epe.cpu().numpy())
        global_edge_all.append(global_edge.cpu().numpy())
        crack_edge_all.append(crack_edge.cpu().numpy())
        fold_all.append(fold.cpu().numpy())
        warp_dice_all.append(warp_dice.cpu().numpy())
        crack_ratio_all.append(crack_ratio.cpu().numpy())

        for j, (img_path, _) in enumerate(chunk):
            per_image.append(
                {
                    "image": img_path.name,
                    "crack_epe_px": float(crack_epe[j].item()),
                    "crack_edge_fidelity": float(crack_edge[j].item()),
                    "warp_crack_dice": float(warp_dice[j].item()),
                    "global_epe_px": float(global_epe[j].item()),
                    "global_edge_fidelity": float(global_edge[j].item()),
                    "folding_rate": float(fold[j].item()),
                    "crack_ratio": float(crack_ratio[j].item()),
                }
            )

    global_epe_np = np.concatenate(global_epe_all, axis=0)
    crack_epe_np = np.concatenate(crack_epe_all, axis=0)
    global_edge_np = np.concatenate(global_edge_all, axis=0)
    crack_edge_np = np.concatenate(crack_edge_all, axis=0)
    fold_np = np.concatenate(fold_all, axis=0)
    warp_dice_np = np.concatenate(warp_dice_all, axis=0)
    crack_ratio_np = np.concatenate(crack_ratio_all, axis=0)

    summary = {
        "method": args.method_name,
        "pred_dir": str(pred_dir),
        "pred_format": args.pred_format,
        "img_dir": str(img_dir),
        "num_samples": int(len(per_image)),
        "primary_crack_epe_px_mean": float(np.mean(crack_epe_np)),
        "primary_crack_epe_px_std": float(np.std(crack_epe_np)),
        "primary_crack_edge_fidelity_mean": float(np.mean(crack_edge_np)),
        "primary_crack_edge_fidelity_std": float(np.std(crack_edge_np)),
        "primary_warp_crack_dice_mean": float(np.mean(warp_dice_np)),
        "primary_warp_crack_dice_std": float(np.std(warp_dice_np)),
        "global_epe_px_mean": float(np.mean(global_epe_np)),
        "global_epe_px_std": float(np.std(global_epe_np)),
        "global_edge_fidelity_mean": float(np.mean(global_edge_np)),
        "global_edge_fidelity_std": float(np.std(global_edge_np)),
        "folding_rate_mean": float(np.mean(fold_np)),
        "folding_rate_std": float(np.std(fold_np)),
        "crack_ratio_mean": float(np.mean(crack_ratio_np)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_summary.json"
    per_image_path = out_dir / "eval_per_image.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "crack_epe_px",
                "crack_edge_fidelity",
                "warp_crack_dice",
                "global_epe_px",
                "global_edge_fidelity",
                "folding_rate",
                "crack_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image)

    print("=== External Flow Evaluation Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"summary saved to: {summary_path}")
    print(f"per-image saved to: {per_image_path}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Evaluate external dense flow/grid predictions")
    parser.add_argument("--pred_dir", required=True, help="外部方法预测 `.npy` 文件目录")
    parser.add_argument("--img_dir", required=True, help="包含图像和 `.npy` 标签的数据目录")
    parser.add_argument("--out_dir", required=True, help="评估结果输出目录")
    parser.add_argument("--method_name", default="external_method", help="外部方法名称，写入 summary")
    parser.add_argument(
        "--pred-format",
        choices=["normalized_grid", "pixel_flow"],
        default="normalized_grid",
        help="预测 `.npy` 的格式；默认与项目标签一致，为归一化逆映射坐标场。",
    )
    parser.add_argument(
        "--pred-suffix",
        default=".npy",
        help="预测文件后缀；默认图像 `xxx.png` 对应 `xxx.png.npy`。",
    )
    parser.add_argument("--size", type=int, default=512, help="评估分辨率")
    parser.add_argument("--batch_size", type=int, default=1, help="批大小")
    parser.add_argument("--num", type=int, default=-1, help="评估样本数，-1 表示全量")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument("--crack_topk", type=float, default=0.08, help="裂缝 ROI 估计 top-k 比例")
    parser.add_argument("--crack_temp", type=float, default=0.07, help="裂缝 ROI 估计温度")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
