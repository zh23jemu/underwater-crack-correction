#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slurm 训练包装入口。

用途：
1. 在不直接修改 `config_crack.py` 的前提下，为集群训练覆盖输出目录、epoch、worker 等参数。
2. 默认关闭 `restart_training`，避免误清空已有的 `output_crackwarp/` 历史 checkpoint 和评估结果。
3. 保持训练主逻辑仍由 `train_v2.py` 负责，后续调模型时不需要维护两套训练代码。

注意：
- 当前 `train_v2.py` 只保存模型权重，没有保存 optimizer/scheduler/scaler 状态，因此严格意义上还不支持
  “从中断位置完整续训”。本脚本主要用于开启新的 Slurm 实验输出目录。
- 如果后续要做真正断点续训，应先扩展 `train_v2.py` 的 checkpoint 保存与加载结构。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import config_crack as config


def parse_args() -> argparse.Namespace:
    """解析 Slurm/命令行传入的训练覆盖参数。"""
    parser = argparse.ArgumentParser(description="CrackWarpNet Slurm 训练包装入口")
    parser.add_argument(
        "--trainroot",
        default=config.trainroot,
        help="训练数据目录，目录内应包含成对的 .png 与 .png.npy 标签。",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="本次实验输出目录；为空时自动使用 output_crackwarp_slurm/时间戳。",
    )
    parser.add_argument("--epochs", type=int, default=config.epochs, help="训练 epoch 数。")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.train_batch_size,
        help="单卡 batch size；当前模型显存压力较大，默认保持 1。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.workers,
        help="DataLoader worker 数，通常与 Slurm 分配 CPU 数接近。",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=config.accum_steps,
        help="梯度累积步数，用于在 batch size 较小时维持等效 batch。",
    )
    parser.add_argument("--lr", type=float, default=config.lr, help="基础学习率。")
    parser.add_argument(
        "--w-crack-mag",
        type=float,
        default=getattr(config, "w_crack_mag", 0.0),
        help="裂缝区域位移幅度一致性损失权重；默认 0 表示关闭。",
    )
    parser.add_argument(
        "--w-crack-edge",
        type=float,
        default=getattr(config, "w_crack_edge", 0.0),
        help="裂缝 ROI 校正图边缘一致性损失权重；默认 0 表示关闭。",
    )
    parser.add_argument(
        "--w-jacobian",
        type=float,
        default=getattr(config, "w_jacobian", 0.0),
        help="坐标场 Jacobian 稳定损失权重；默认 0 表示关闭。",
    )
    parser.add_argument(
        "--w-crack-coord-extra",
        type=float,
        default=getattr(config, "w_crack_coord_extra", 0.0),
        help="高置信裂缝核心坐标额外监督权重；默认 0 表示关闭。",
    )
    parser.add_argument(
        "--crack-mag-robust-delta",
        type=float,
        default=getattr(config, "crack_mag_robust_delta", 0.0),
        help="位移幅度一致性的 Huber 阈值；0 表示沿用普通 Charbonnier。",
    )
    parser.add_argument(
        "--crack-mag-over-weight",
        type=float,
        default=getattr(config, "crack_mag_over_weight", 0.0),
        help="预测位移幅度超过 GT 时的额外惩罚权重；0 表示关闭。",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=getattr(config, "init_checkpoint", ""),
        help="可选：从已有模型权重初始化本次训练，适合 loss 小改动 fine-tune。",
    )
    parser.add_argument(
        "--hard-sample-list",
        default=getattr(config, "hard_sample_list", ""),
        help="可选：困难样本清单；支持文件名、无后缀样本名或原始裂缝族群编号。",
    )
    parser.add_argument(
        "--hard-sample-weight",
        type=float,
        default=getattr(config, "hard_sample_weight", 1.0),
        help="困难样本采样权重；大于 1 时启用 WeightedRandomSampler。",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="进程内可见 GPU 编号；Slurm 单卡任务中通常使用 0。",
    )
    return parser.parse_args()


def main() -> None:
    """覆盖配置后调用原始训练入口。"""
    args = parse_args()

    # 自动生成独立实验目录，避免多次 Slurm 提交互相覆盖，也避免触碰历史输出。
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output_crackwarp_slurm") / stamp

    # 仅覆盖必要配置，其余损失权重、模型结构和数据增强策略仍沿用 config_crack.py。
    config.trainroot = args.trainroot
    config.output_dir = str(output_dir)
    config.epochs = args.epochs
    config.train_batch_size = args.batch_size
    config.workers = args.workers
    config.accum_steps = args.accum_steps
    config.lr = args.lr
    config.w_crack_mag = args.w_crack_mag
    config.w_crack_edge = args.w_crack_edge
    config.w_jacobian = args.w_jacobian
    config.w_crack_coord_extra = args.w_crack_coord_extra
    config.crack_mag_robust_delta = args.crack_mag_robust_delta
    config.crack_mag_over_weight = args.crack_mag_over_weight
    config.init_checkpoint = args.init_checkpoint
    config.hard_sample_list = args.hard_sample_list
    config.hard_sample_weight = args.hard_sample_weight
    config.gpu_id = args.gpu_id

    # 重要保护：当前原训练脚本中 restart_training=True 会删除 output_dir。
    # Slurm 默认应保留历史结果，因此这里强制关闭；如需全新实验，请直接换新的 output_dir。
    config.restart_training = False

    print("[Slurm wrapper] trainroot =", config.trainroot)
    print("[Slurm wrapper] output_dir =", config.output_dir)
    print("[Slurm wrapper] epochs =", config.epochs)
    print("[Slurm wrapper] batch_size =", config.train_batch_size)
    print("[Slurm wrapper] workers =", config.workers)
    print("[Slurm wrapper] accum_steps =", config.accum_steps)
    print("[Slurm wrapper] lr =", config.lr)
    print("[Slurm wrapper] w_crack_mag =", config.w_crack_mag)
    print("[Slurm wrapper] w_crack_edge =", config.w_crack_edge)
    print("[Slurm wrapper] w_jacobian =", config.w_jacobian)
    print("[Slurm wrapper] w_crack_coord_extra =", config.w_crack_coord_extra)
    print("[Slurm wrapper] crack_mag_robust_delta =", config.crack_mag_robust_delta)
    print("[Slurm wrapper] crack_mag_over_weight =", config.crack_mag_over_weight)
    print("[Slurm wrapper] init_checkpoint =", config.init_checkpoint)
    print("[Slurm wrapper] hard_sample_list =", config.hard_sample_list)
    print("[Slurm wrapper] hard_sample_weight =", config.hard_sample_weight)
    print("[Slurm wrapper] restart_training =", config.restart_training)

    # 在配置覆盖完成后再导入训练脚本，确保 train_v2.py 拿到的是同一个 config_crack 模块实例。
    import train_v2

    train_v2.main()


if __name__ == "__main__":
    main()
