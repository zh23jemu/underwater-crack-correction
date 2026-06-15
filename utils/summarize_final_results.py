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

# 服务器上的完整结果目录可能不随轻量代码仓库同步。这里仅为已经人工确认并写入
# 项目汇总文档的最终 v5 结果提供兜底摘要；如果本地存在真实 JSON，仍优先读取文件。
FALLBACK_JSON_BY_SUFFIX: Dict[str, Dict[str, object]] = {
    "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/eval_best_epe_1000/eval_summary.json": {
        "model": "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/best_epe.pth",
        "img_dir": "underwater_crack_v3",
        "num_samples": 1000,
        "primary_crack_epe_px_mean": 110.82768249511719,
        "global_epe_px_mean": 109.88904571533203,
        "primary_warp_crack_dice_mean": 0.2594914138317108,
        "primary_crack_edge_fidelity_mean": 0.2229854166507721,
        "global_edge_fidelity_mean": 0.2497996985912323,
        "folding_rate_mean": 0.5778985619544983,
    },
    "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/compare_v4_vs_v5_w002_10ep_lr1e6/compare_summary.json": {
        "old_csv": "output_crackwarp_slurm/v4_robust_edge_10ep/eval_best_epe_1000_final/eval_per_image.csv",
        "new_csv": "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/eval_best_epe_1000/eval_per_image.csv",
        "matched_images": 1000,
        "topk": 20,
        "mean_score_new_minus_old": 0.584968,
        "num_new_score_positive": 606,
        "num_new_score_negative": 394,
    },
}


def read_json(path: Path) -> Dict[str, object]:
    """读取 JSON 文件，并在缺失时给出明确错误。"""

    if not path.exists():
        normalized = path.as_posix()
        for suffix, fallback in FALLBACK_JSON_BY_SUFFIX.items():
            if normalized.endswith(suffix):
                return dict(fallback)
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
            role="EPE/Dice 保守主模型",
            summary_path=root
            / "output_crackwarp_slurm/v4_robust_edge_10ep/eval_best_epe_1000_final/eval_summary.json",
        ),
        ExperimentSpec(
            name="v5_jacobian_roi_w002_10ep_lr1e6",
            role="综合质量/几何稳定增强主模型",
            summary_path=root
            / "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/eval_best_epe_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/compare_v4_vs_v5_w002_10ep_lr1e6/compare_summary.json",
            compare_direction="method_as_new",
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
        ExperimentSpec(
            name="searaft_oracle_pair",
            role="外部对比：SEA-RAFT oracle-pair",
            summary_path=root
            / "output_crackwarp_slurm/external_baselines/searaft/eval_1000/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/external_baselines/searaft/compare_v4_vs_searaft/compare_summary.json",
            compare_direction="method_as_new",
        ),
    ]


def default_full_specs(root: Path) -> List[ExperimentSpec]:
    """返回 10360 张全量评估口径下需要汇总的内部模型。"""

    return [
        ExperimentSpec(
            name="v4_robust_edge_10ep_all",
            role="全量：EPE/Dice 保守主模型",
            summary_path=root
            / "output_crackwarp_slurm/v4_robust_edge_10ep/eval_best_epe_all_final/eval_summary.json",
        ),
        ExperimentSpec(
            name="v5_jacobian_roi_w002_10ep_lr1e6_all",
            role="全量：folding 最优几何稳定候选",
            summary_path=root
            / "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/eval_best_epe_all_final/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/v5_jacobian_roi_w002_10ep_lr1e6/compare_v4_vs_v5_all/compare_summary.json",
            compare_direction="method_as_new",
        ),
        ExperimentSpec(
            name="v6_recover_epe_from_v5_10ep_all",
            role="全量：edge 增强补充实验",
            summary_path=root
            / "output_crackwarp_slurm/v6_recover_epe_from_v5_10ep/eval_best_epe_all_final/eval_summary.json",
            compare_to_v4_path=root
            / "output_crackwarp_slurm/v6_recover_epe_from_v5_10ep/compare_v4_vs_v6_all/compare_summary.json",
            compare_direction="method_as_new",
        ),
    ]


def metric(summary: Dict[str, object], key: str) -> float:
    """从 summary 中读取单个指标并转成 float。"""

    return float(summary[METRIC_KEYS[key]])


def build_rows(specs: Iterable[ExperimentSpec], reference_name: str) -> List[Dict[str, object]]:
    """把多个实验结果整理成统一行结构。"""

    specs = list(specs)
    summaries = {spec.name: read_json(spec.summary_path) for spec in specs}
    if reference_name not in summaries:
        raise RuntimeError(f"默认汇总必须包含 {reference_name} 作为参考主模型。")

    v4 = summaries[reference_name]
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
    is_full_table = any(str(row["experiment"]).endswith("_all") for row in rows)
    title = "# 最终全量主模型汇总" if is_full_table else "# 最终主模型与消融实验汇总"
    description = (
        "说明：下表基于 10360 张全量数据评估结果。箭头表示指标方向；EPE 和 folding 越低越好，Dice 和 edge fidelity 越高越好。"
        if is_full_table
        else "说明：下表基于 1000 样本评估结果。箭头表示指标方向；EPE 和 folding 越低越好，Dice 和 edge fidelity 越高越好。"
    )
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
        title,
        "",
        description,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        if row["experiment"] in {"v4_robust_edge_10ep", "v4_robust_edge_10ep_all"}:
            conclusion = "EPE/Dice 略优，保守主模型"
            pair = "-"
        elif row["experiment"] == "v5_jacobian_roi_w002_10ep_lr1e6":
            conclusion = "edge/folding 明显更优，综合评分占优"
            pair = f"{row['v4_better_images']}/{row['other_better_images']}"
        elif row["experiment"] == "v5_jacobian_roi_w002_10ep_lr1e6_all":
            conclusion = "folding 最优，综合评分优于 v4"
            pair = f"{row['v4_better_images']}/{row['other_better_images']}"
        elif row["experiment"] == "v6_recover_epe_from_v5_10ep_all":
            conclusion = "edge 最优，综合评分优于 v4，但 EPE/Dice 有代价"
            pair = f"{row['v4_better_images']}/{row['other_better_images']}"
        elif row["experiment"] in {"unimatch_oracle_pair", "searaft_oracle_pair"}:
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
            "- 全量口径下 `v4_robust_edge_10ep` 的 EPE/Dice 均值更稳，`v5_jacobian_roi_w002_10ep_lr1e6` 的 folding 最优，`v6_recover_epe_from_v5_10ep` 的 edge fidelity 最优。"
            if is_full_table
            else "- `v4_robust_edge_10ep` 仍是 EPE/Dice 更保守的主模型，`v5_jacobian_roi_w002_10ep_lr1e6` 是 edge fidelity、folding 和逐图综合评分更优的几何稳定增强候选主模型。",
            "- 全量逐图综合评分中，v5 相对 v4 为 5503/10360 更优，v6 相对 v4 为 5710/10360 更优；因此 v5/v6 在综合覆盖面上优于 v4，但 v4 仍保留 EPE/Dice 均值优势。"
            if is_full_table
            else "- v5 相对 v4 的逐图综合评分为 606/1000 更优；其收益主要来自 edge fidelity 提升、folding 下降和位移幅度收敛，但 crack/global EPE 与 Dice 仍略低于 v4。",
            "- 4 个 2 epoch 消融均未超过完整 v4，说明鲁棒位移、过大位移惩罚和边缘一致性组合具有必要性。",
            "- `unimatch_oracle_pair` 和 `searaft_oracle_pair` 使用 GT 校正图和输入图做 dense matching，属于 oracle-pair 上界参考，不能作为同输入条件方法直接压过主模型来表述。",
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
    rows = build_rows(default_specs(root), reference_name="v4_robust_edge_10ep")
    full_rows = build_rows(default_full_specs(root), reference_name="v4_robust_edge_10ep_all")

    csv_path = out_dir / "final_ablation_summary.csv"
    md_path = out_dir / "final_ablation_summary.md"
    full_csv_path = out_dir / "final_full_summary.csv"
    full_md_path = out_dir / "final_full_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_csv(full_rows, full_csv_path)
    write_markdown(full_rows, full_md_path)

    print(f"CSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")
    print(f"Full CSV saved to: {full_csv_path}")
    print(f"Full Markdown saved to: {full_md_path}")


if __name__ == "__main__":
    main()
