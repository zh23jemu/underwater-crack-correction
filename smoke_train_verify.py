#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本机最小训练验证脚本。

用途：
1. 验证新建 `.venv` 后，PyTorch、OpenCV、数据读取、模型前向、损失函数、反向传播和验证流程是否能跑通。
2. 默认只使用极少量样本，并把图像和标签缩放到 128×128，避免本机小显存 GPU 直接跑 512×512 全模型。
3. 不使用 `train_v2.py` 的 `restart_training` 逻辑，因此不会清空已有 `output_crackwarp/` 历史结果。

注意：
- 这是 smoke test，不用于评估模型最终效果。
- 正式训练仍建议使用 `slurm_train_crackwarp.sbatch` 提交到 Slurm。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from loss_crack import CrackWarpLoss
from models.crack_warp_net import CrackWarpNet


class TinyCrackDataset(Dataset):
    """只读取少量成对 `.png` / `.png.npy` 样本的最小数据集。"""

    def __init__(self, root: str, size: int, limit: int) -> None:
        self.root = Path(root)
        self.size = int(size)

        # 仅保留存在配套标签的图片，避免 smoke test 被坏样本干扰。
        pairs: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.root.glob("*.png")):
            label_path = Path(str(img_path) + ".npy")
            if label_path.exists():
                pairs.append((img_path, label_path))
            if len(pairs) >= limit:
                break

        if not pairs:
            raise RuntimeError(f"未在 {self.root} 找到成对的 .png / .png.npy 样本")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label_path = self.pairs[index]

        # OpenCV 读取 BGR，再转 RGB，保持与训练主链路的张量格式一致。
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"读取图片失败：{img_path}")
        img_bgr = cv2.resize(img_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        image = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()

        # 标签是归一化逆映射坐标场，缩放空间尺寸后仍保持 [0,1] 坐标定义。
        label_np = np.load(label_path).astype(np.float32)
        label_np = np.stack(
            [
                cv2.resize(label_np[0], (self.size, self.size), interpolation=cv2.INTER_LINEAR),
                cv2.resize(label_np[1], (self.size, self.size), interpolation=cv2.INTER_LINEAR),
            ],
            axis=0,
        )
        label_np = np.clip(label_np, 0.0, 1.0)
        label = torch.from_numpy(label_np).float()
        return image, label


def compute_epe_px(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算像素尺度 EPE；缩放尺寸为 S 时，归一化坐标乘以 S-1。"""
    scale = float(pred.shape[-1] - 1)
    diff = (pred - target) * scale
    epe = torch.sqrt((diff ** 2).sum(dim=1))
    return float(epe.mean().detach().cpu())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本机最小训练验证")
    parser.add_argument("--data-root", default="underwater_crack_v3", help="训练样本目录")
    parser.add_argument("--output-dir", default="output_smoke_local", help="smoke 输出目录")
    parser.add_argument("--size", type=int, default=128, help="smoke 输入尺寸")
    parser.add_argument("--samples", type=int, default=4, help="读取样本数")
    parser.add_argument("--epochs", type=int, default=1, help="最小训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="smoke 学习率")
    parser.add_argument("--base-ch", type=int, default=8, help="轻量模型基础通道数")
    parser.add_argument("--num-heads", type=int, default=2, help="轻量 Transformer 注意力头数")
    parser.add_argument("--n-iter", type=int, default=1, help="迭代细化次数")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"[smoke] device={device}")
    if use_cuda:
        print(f"[smoke] gpu={torch.cuda.get_device_name(0)}")

    dataset = TinyCrackDataset(args.data_root, size=args.size, limit=args.samples)
    train_count = max(1, len(dataset) - 1)
    val_count = len(dataset) - train_count
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 使用轻量网络验证训练链路，避免本机小显存 GPU 跑完整 512 模型。
    # 这里保留项目真实 CrackWarpNet 结构，但显著降低通道数和迭代次数。
    model = CrackWarpNet(
        base_ch=args.base_ch,
        num_heads=args.num_heads,
        n_iter=args.n_iter,
        n_transformer_blocks=1,
        drop_out_rate=0.0,
    ).to(device)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[smoke] tiny_model_params={params_m:.3f}M")

    criterion = CrackWarpLoss(
        w_coord=1.0,
        w_smooth=0.05,
        w_fold=0.01,
        w_photo=0.0,
        w_ssim=0.0,
        w_freq=0.0,
        w_crack_coord=0.2,
        w_crack_grad=0.0,
        w_crack_freq=0.0,
        crack_boost=2.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        losses: List[float] = []
        epes: List[float] = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            flows = model(images)
            loss, loss_dict = criterion(flows, labels, img=images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            final_flow = flows[-1] if isinstance(flows, list) else flows
            losses.append(float(loss.detach().cpu()))
            epes.append(compute_epe_px(final_flow, labels))
        print(
            f"[smoke] epoch={epoch + 1}/{args.epochs} "
            f"train_loss={np.mean(losses):.6f} train_epe={np.mean(epes):.3f}px"
        )

    model.eval()
    val_losses: List[float] = []
    val_epes: List[float] = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            flows = model(images)
            loss, _ = criterion(flows, labels, img=images)
            final_flow = flows[-1] if isinstance(flows, list) else flows
            val_losses.append(float(loss.detach().cpu()))
            val_epes.append(compute_epe_px(final_flow, labels))

    ckpt_path = out_dir / "smoke_tiny_model.pth"
    torch.save(model.state_dict(), ckpt_path)

    print(
        f"[smoke] val_loss={np.mean(val_losses):.6f} "
        f"val_epe={np.mean(val_epes):.3f}px"
    )
    print(f"[smoke] saved={ckpt_path}")
    print("[smoke] OK: 最小训练和验证链路已跑通")


if __name__ == "__main__":
    main()
