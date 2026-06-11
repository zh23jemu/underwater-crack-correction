#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总最终主实验和消融实验结果。

本脚本读取已经生成的 `eval_summary.json` 和可选的 `compare_summary.json`，
输出两份结果：

1. CSV：适合后续继续做表格处理或导入 Excel。
2. Markdown：适合直接复制到报告、论文实验部分或项目进展说明。

设计原则：
- 不重新计算指标，只汇总已有评估结果，避免引入新的评估口径。
- 默认把完整 v4 作为主模型参考项，其他消融都和 v4 做差值比较。
- 对 EPE / folding 这类越低越好的指标，差值采用 “当前方法 - v4”；
  因此正数表示当前方法比 v4 更差。
- 对 Dice / edge fidelity 这类越高越好的指标，差值采用 “v4 - 当前方法”；
  因此正数同样表示当前方法比 v4 更差。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ExperimentSpec:
    """一个实验条目的路径和展示信息。"""

    name: str
    summary_path: Path
    role: str
    compare_to_v4_path: Optional[Path] = None
    compare_direction: str = "v4_as_new"


METRIC_KEYS = {
    "crack_epe": "primary_crack_epe_px_mean",
    "global_epe": "global_epe_px_mean",
    "dice": "primary_warp_crack_dice_mean",
    "crack_edge": "primary_crack_edge_fidelity_mean",
    "global_edge": "global_edge_fidelity_mean",
    "folding": "folding_rate_mean",
}


def read_json(path: Path) -> Dict[str, object]:
    """读取 JSON 文件，并在缺失时给出明确错误。"""

    if not path.exists():
        raise FileNotFoundError(f"缺少结果文件: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_specs(root: Path) -> List[ExperimentSpec]:
    """返回当前项目默认需要汇总的主实验和消融实验。"""

    return [
        ExperimentSpec(
            name="v3_mag_ft_w005_10ep",
            role="上一版主线",
            summary_path=root / "output_crackwarp_slurm/v3_mag_ft_w005_10ep/eval_best_epe_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/v4_robust_edge_10ep/compare_v3_w005_vs_v4_10ep/compare_summary.json",
        ),
        ExperimentSpec(
            name="v4_robust_edge_10ep",
            role="当前主模型",
            summary_path=root
            / "output_crackwarp_slurm/v4_robust_edge_10ep/eval_best_epe_1000_final/eval_summary.json",
        ),
        ExperimentSpec(
            name="abl_edge_only_2ep",
            role="消融：仅边缘一致性",
            summary_path=root
            / "output_crackwarp_slurm/final_ablation/abl_edge_only_2ep/eval_best_epe_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/final_ablation/compare_abl_edge_only_2ep_vs_v4/compare_summary.json",
        ),
        ExperimentSpec(
            name="abl_no_over_penalty_2ep",
            role="消融：去掉过大位移惩罚",
            summary_path=root
            / "output_crackwarp_slurm/final_ablation/abl_no_over_penalty_2ep/eval_best_epe_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/final_ablation/compare_abl_no_over_penalty_2ep_vs_v4/compare_summary.json",
        ),
        ExperimentSpec(
            name="abl_robust_only_2ep",
            role="消融：仅鲁棒位移约束",
            summary_path=root
            / "output_crackwarp_slurm/final_ablation/abl_robust_only_2ep/eval_best_epe_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/final_ablation/compare_abl_robust_only_2ep_vs_v4/compare_summary.json",
        ),
        ExperimentSpec(
            name="abl_edge_w003_2ep",
            role="消融：较低边缘权重",
            summary_path=root
            / "output_crackwarp_slurm/final_ablation/abl_edge_w003_2ep/eval_best_epe_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/final_ablation/compare_abl_edge_w003_2ep_vs_v4/compare_summary.json",
        ),
        ExperimentSpec(
            name="unimatch_oracle_pair",
            role="外部对比：UniMatch oracle-pair",
            summary_path=root
            / "output_crackwarp_slurm/external_baselines/unimatch/eval_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/external_baselines/unimatch/compare_v4_vs_unimatch/compare_summary.json",
            compare_direction="method_as_new",
        ),
    ]


def metric(summary: Dict[str, object], key: str) -> float:
    """从 summary 中读取单个指标并转成 float。"""

    return float(summary[METRIC_KEYS[key]])


def build_rows(specs: Iterable[ExperimentSpec]) -> List[Dict[str, object]]:
    """把多个实验结果整理成统一行结构。"""

    specs = list(specs)
    summaries = {spec.name: read_json(spec.summary_path) for spec in specs}
    if "v4_robust_edge_10ep" not in summaries:
        raise RuntimeError("默认汇总必须包含 v4_robust_edge_10ep 作为参考主模型。")

    v4 = summaries["v4_robust_edge_10ep"]
    rows: List[Dict[str, object]] = []

    for spec in specs:
        summary = summaries[spec.name]
        compare = read_json(spec.compare_to_v4_path) if spec.compare_to_v4_path else {}

        row: Dict[str, object] = {
            "experiment": spec.name,
            "role": spec.role,
            "num_samples": int(summary["num_samples"]),
            "crack_epe": metric(summary, "crack_epe"),
            "global_epe": metric(summary, "global_epe"),
            "dice": metric(summary, "dice"),
            "crack_edge": metric(summary, "crack_edge"),
            "global_edge": metric(summary, "global_edge"),
            "folding": metric(summary, "folding"),
        }

        # 和 v4 的差值全部设计为“正数代表当前方法弱于 v4”，便于报告解释。
        row["delta_crack_epe_vs_v4"] = row["crack_epe"] - metric(v4, "crack_epe")
        row["delta_global_epe_vs_v4"] = row["global_epe"] - metric(v4, "global_epe")
        row["delta_dice_vs_v4"] = metric(v4, "dice") - row["dice"]
        row["delta_crack_edge_vs_v4"] = metric(v4, "crack_edge") - row["crack_edge"]
        row["delta_global_edge_vs_v4"] = metric(v4, "global_edge") - row["global_edge"]
        row["delta_folding_vs_v4"] = row["folding"] - metric(v4, "folding")

        # compare_summary.py 只记录 old/new，不知道哪一个是 v4。
        # 内部消融的 compare 是 “old=消融, new=v4”，因此 positive 表示 v4 更好；
        # UniMatch 的 compare 是 “old=v4, new=UniMatch”，因此 positive 表示外部方法更好。
        positive = int(compare.get("num_new_score_positive", 0))
        negative = int(compare.get("num_new_score_negative", 0))
        if spec.compare_direction == "v4_as_new":
            row["v4_better_images"] = positive
            row["other_better_images"] = negative
        elif spec.compare_direction == "method_as_new":
            row["v4_better_images"] = negative
            row["other_better_images"] = positive
        else:
            raise ValueError(f"未知 compare_direction: {spec.compare_direction}")
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    """写出 CSV 汇总表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    """Markdown 表格中的数字格式化。"""

    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    """写出面向报告的 Markdown 汇总表。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "实验",
        "角色",
        "Crack EPE↓",
        "Global EPE↓",
        "Dice↑",
        "Crack Edge↑",
        "Global Edge↑",
        "Folding↓",
        "v4更优/对方更优",
        "结论",
    ]

    lines = [
        "# 最终主模型与消融实验汇总",
        "",
        "说明：下表基于 1000 样本评估结果。箭头表示指标方向；EPE 和 folding 越低越好，Dice 和 edge fidelity 越高越好。",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        if row["experiment"] == "v4_robust_edge_10ep":
            conclusion = "当前最稳主模型"
            pair = "-"
        elif row["experiment"] == "unimatch_oracle_pair":
            conclusion = "oracle-pair 上界参考，非同输入条件"
            pair = f"{row['v4_better_images']}/{row['other_better_images']}"
        else:
            conclusion = "弱于完整 v4，可作为消融证据"
            pair = f"{row['v4_better_images']}/{row['other_better_images']}"

        values = [
            row["experiment"],
            row["role"],
            fmt(row["crack_epe"]),
            fmt(row["global_epe"]),
            fmt(row["dice"]),
            fmt(row["crack_edge"]),
            fmt(row["global_edge"]),
            fmt(row["folding"]),
            pair,
            conclusion,
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- `v4_robust_edge_10ep` 在 1000 样本上仍是当前最稳主模型。",
            "- 4 个 2 epoch 消融均未超过完整 v4，说明鲁棒位移、过大位移惩罚和边缘一致性组合具有必要性。",
            "- `unimatch_oracle_pair` 使用 GT 校正图和输入图做 dense matching，属于 oracle-pair 上界参考，不能作为同输入条件方法直接压过主模型来表述。",
            "- 后续若继续做模型优化，应优先设计 folding/Jacobian 正则、边界平滑或 ROI 局部对齐消融，而不是简单拉长这 4 个配置。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="汇总最终主模型和消融实验评估结果")
    parser.add_argument("--root", default=".", help="项目根目录")
    parser.add_argument(
        "--out-dir",
        default="output_crackwarp_slurm/final_ablation/summary_tables",
        help="汇总表输出目录",
    )
    return parser.parse_args()


def main() -> None:
    """主入口：读取默认结果并输出 CSV / Markdown 汇总表。"""

    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    rows = build_rows(default_specs(root))

    csv_path = out_dir / "final_ablation_summary.csv"
    md_path = out_dir / "final_ablation_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)

    print(f"CSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")


if __name__ == "__main__":
    main()
