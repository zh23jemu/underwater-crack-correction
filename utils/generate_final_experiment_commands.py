#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成最终阶段实验命令。

这个脚本不直接提交 Slurm 任务，只负责把主实验、消融实验和评估命令统一打印出来。
这样做有两个好处：
1. 避免手写环境变量时漏参数，保证不同实验之间只有预期变量不同。
2. 方便先审阅命令，再复制到服务器执行，降低覆盖历史结果的风险。

使用示例：
    .venv/bin/python utils/generate_final_experiment_commands.py --stage all
    .venv/bin/python utils/generate_final_experiment_commands.py --stage ablation --epochs 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class AblationExperiment:
    """单个消融实验的参数集合。"""

    name: str
    init_checkpoint: str
    w_crack_mag: float
    w_crack_edge: float
    crack_mag_robust_delta: float
    crack_mag_over_weight: float
    note: str


def shell_quote(value: str) -> str:
    """为 shell 命令做最小转义，避免路径中出现特殊字符时破坏命令。"""

    escaped = value.replace("'", "'\"'\"'")
    return f"'{escaped}'"


def build_sbatch_command(
    *,
    output_dir: str,
    epochs: int,
    lr: str,
    partition: str,
    time_limit: str,
    init_checkpoint: str = "",
    w_crack_mag: float = 0.0,
    w_crack_edge: float = 0.0,
    crack_mag_robust_delta: float = 0.0,
    crack_mag_over_weight: float = 0.0,
) -> str:
    """拼接 Slurm 提交命令。

    参数只覆盖当前实验真正需要变化的部分，其余资源配置继续沿用
    `slurm_train_crackwarp.sbatch` 内部默认值，避免命令过长且难维护。
    """

    env_parts = [
        f"OUTPUT_DIR={shell_quote(output_dir)}",
        f"EPOCHS={epochs}",
        f"LR={lr}",
        f"W_CRACK_MAG={w_crack_mag}",
        f"W_CRACK_EDGE={w_crack_edge}",
        f"CRACK_MAG_ROBUST_DELTA={crack_mag_robust_delta}",
        f"CRACK_MAG_OVER_WEIGHT={crack_mag_over_weight}",
    ]
    if init_checkpoint:
        env_parts.append(f"INIT_CHECKPOINT={shell_quote(init_checkpoint)}")

    return (
        " ".join(env_parts)
        + f" sbatch --partition={partition} --time={time_limit} slurm_train_crackwarp.sbatch"
    )


def build_eval_command(model_path: str, out_dir: str, num_samples: int) -> str:
    """生成统一评估命令，默认使用项目本地虚拟环境。"""

    return (
        ".venv/bin/python utils/evaluate_metrics.py "
        f"--model {shell_quote(model_path)} "
        "--img_dir underwater_crack_v3 "
        f"--out_dir {shell_quote(out_dir)} "
        f"--num {num_samples} "
        "--batch_size 1 "
        "--gpu 0"
    )


def default_ablations(v3_checkpoint: str) -> List[AblationExperiment]:
    """返回当前最小消融矩阵。

    这些消融围绕 v4 的核心改动展开：裂缝位移幅度校准、鲁棒化、
    过大位移惩罚和裂缝边缘一致性。默认都从 v3 主线 checkpoint
    初始化，便于公平比较单个改动的贡献。
    """

    return [
        AblationExperiment(
            name="abl_robust_only",
            init_checkpoint=v3_checkpoint,
            w_crack_mag=0.05,
            crack_mag_robust_delta=0.01,
            crack_mag_over_weight=0.5,
            w_crack_edge=0.0,
            note="只验证鲁棒位移幅度一致性和过大位移惩罚，不加入边缘一致性。",
        ),
        AblationExperiment(
            name="abl_edge_only",
            init_checkpoint=v3_checkpoint,
            w_crack_mag=0.0,
            crack_mag_robust_delta=0.0,
            crack_mag_over_weight=0.0,
            w_crack_edge=0.05,
            note="只验证裂缝 ROI 边缘一致性，观察是否单独改善 edge fidelity。",
        ),
        AblationExperiment(
            name="abl_no_over_penalty",
            init_checkpoint=v3_checkpoint,
            w_crack_mag=0.05,
            crack_mag_robust_delta=0.01,
            crack_mag_over_weight=0.0,
            w_crack_edge=0.05,
            note="去掉过大位移额外惩罚，验证该项对高难样本 p95 位移和 folding 的作用。",
        ),
        AblationExperiment(
            name="abl_edge_w003",
            init_checkpoint=v3_checkpoint,
            w_crack_mag=0.05,
            crack_mag_robust_delta=0.01,
            crack_mag_over_weight=0.5,
            w_crack_edge=0.03,
            note="使用更保守的边缘损失权重，观察能否降低少数 ROI 对齐退化。",
        ),
        AblationExperiment(
            name="abl_v4_full_repeat",
            init_checkpoint=v3_checkpoint,
            w_crack_mag=0.05,
            crack_mag_robust_delta=0.01,
            crack_mag_over_weight=0.5,
            w_crack_edge=0.05,
            note="完整 v4 配置复现实验；若已有 v4_robust_edge_10ep，可优先不重复训练。",
        ),
    ]


def print_section(title: str) -> None:
    """打印 Markdown 风格的小标题，方便复制到记录文档。"""

    print(f"\n## {title}\n")


def emit_main_eval(args: argparse.Namespace) -> None:
    """输出最终主模型评估命令。"""

    print_section("主模型最终评估")
    print("# 1000 样本主评估")
    print(build_eval_command(args.v4_checkpoint, "output_crackwarp_slurm/v4_robust_edge_10ep/eval_best_epe_1000_final", 1000))
    print()
    print("# 全量评估；如果服务器时间紧，可以先跳过")
    print(build_eval_command(args.v4_checkpoint, "output_crackwarp_slurm/v4_robust_edge_10ep/eval_best_epe_all", -1))


def emit_ablation(args: argparse.Namespace) -> None:
    """输出消融实验 Slurm 命令。"""

    print_section("消融实验训练")
    for exp in default_ablations(args.v3_checkpoint):
        output_dir = f"output_crackwarp_slurm/final_ablation/{exp.name}_{args.epochs}ep"
        print(f"# {exp.name}: {exp.note}")
        print(
            build_sbatch_command(
                output_dir=output_dir,
                epochs=args.epochs,
                lr=args.lr,
                partition=args.partition,
                time_limit=args.time,
                init_checkpoint=exp.init_checkpoint,
                w_crack_mag=exp.w_crack_mag,
                w_crack_edge=exp.w_crack_edge,
                crack_mag_robust_delta=exp.crack_mag_robust_delta,
                crack_mag_over_weight=exp.crack_mag_over_weight,
            )
        )
        print()


def emit_ablation_eval(args: argparse.Namespace) -> None:
    """输出消融实验完成后的统一评估命令。"""

    print_section("消融实验评估")
    for exp in default_ablations(args.v3_checkpoint):
        output_dir = f"output_crackwarp_slurm/final_ablation/{exp.name}_{args.epochs}ep"
        model_path = f"{output_dir}/best_epe.pth"
        eval_dir = f"{output_dir}/eval_best_epe_1000"
        print(f"# {exp.name}")
        print(build_eval_command(model_path, eval_dir, 1000))
        print()


def emit_external_baselines() -> None:
    """输出外部对比方法的执行占位清单。

    外部方法通常需要先 clone 代码、下载权重、适配输出格式，因此这里不生成
    直接可运行命令，而是固定优先级和适配要求，防止后续对比实验范围漂移。
    """

    print_section("外部对比方法优先级")
    rows = [
        ("A", "SEA-RAFT / SEA-RAFT-M", "2024 光流方法；优先适配，输出 flow 后转换为当前归一化逆映射坐标。"),
        ("A", "UniMatch", "2023 统一匹配/光流方法；代码成熟，适合作为强基线。"),
        ("B", "MemFlow", "2024 记忆光流；若当前数据能组织伪序列，可作为扩展对比。"),
        ("B", "NeuFlow / NeuFlow v2", "2024-2025 轻量快速光流；适合补速度和效率对比。"),
        ("C", "RAFT / GMFlow / FlowFormer", "作为历史强基线或补充，不建议作为唯一对比。"),
    ]
    for priority, name, note in rows:
        print(f"- [{priority}] {name}: {note}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="生成最终主实验、消融实验和对比实验命令")
    parser.add_argument(
        "--stage",
        choices=["all", "main", "ablation", "ablation-eval", "external"],
        default="all",
        help="需要输出的命令阶段。",
    )
    parser.add_argument(
        "--v3-checkpoint",
        default="output_crackwarp_slurm/v3_mag_ft_w005_10ep/best_epe.pth",
        help="消融实验默认初始化权重，通常使用 v3 主线 best_epe。",
    )
    parser.add_argument(
        "--v4-checkpoint",
        default="output_crackwarp_slurm/v4_robust_edge_10ep/best_epe.pth",
        help="最终主模型候选权重。",
    )
    parser.add_argument("--epochs", type=int, default=10, help="消融 fine-tune epoch 数。")
    parser.add_argument("--lr", default="3e-6", help="消融 fine-tune 学习率。")
    parser.add_argument("--partition", default="gpu", help="Slurm 分区；短任务可手动改为 gpuHz。")
    parser.add_argument("--time", default="24:00:00", help="Slurm 时间限制。")
    return parser.parse_args(argv)


def main() -> None:
    """主入口：按阶段打印命令。"""

    args = parse_args()
    if args.stage in {"all", "main"}:
        emit_main_eval(args)
    if args.stage in {"all", "ablation"}:
        emit_ablation(args)
    if args.stage in {"all", "ablation-eval"}:
        emit_ablation_eval(args)
    if args.stage in {"all", "external"}:
        emit_external_baselines()


if __name__ == "__main__":
    main()
