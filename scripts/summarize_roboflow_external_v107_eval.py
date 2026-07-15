#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 Roboflow 两个外部候选评估集上的外部四模型 + v107 指标。

该脚本只读取已经生成的 eval_summary.json，不重新计算指标。缺失项会进入状态
Markdown，便于判断 Word 表格中哪些位置仍不能填写真实数据。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_KEYS = {
    "crack_epe": "primary_crack_epe_px_mean",
    "global_epe": "global_epe_px_mean",
    "dice": "primary_warp_crack_dice_mean",
    "crack_edge": "primary_crack_edge_fidelity_mean",
    "global_edge": "global_edge_fidelity_mean",
    "folding": "folding_rate_mean",
}


@dataclass(frozen=True)
class DatasetSpec:
    """需要汇总的一个 Roboflow 评估集。"""

    key: str
    label: str


@dataclass(frozen=True)
class MethodSpec:
    """表格中的一个方法。"""

    name: str
    subdir: str
    method_type: str
    note: str


DATASETS = [
    DatasetSpec("roboflow_underwater_crack", "Roboflow underwater crack"),
    DatasetSpec("roboflow_concrete_blue_crack", "Roboflow concrete / blue crack"),
]

METHODS = [
    MethodSpec("RAFT", "raft", "外部 oracle-pair", "torchvision RAFT-large"),
    MethodSpec("UniMatch", "unimatch", "外部 oracle-pair", "GMFlow scale2 reg-refine6 mixed-data"),
    MethodSpec("SEA-RAFT", "searaft", "外部 oracle-pair", "SEA-RAFT 公开预训练配置"),
    MethodSpec("GMA/RAFT-small fallback", "gma", "外部 oracle-pair", "GMA 权重缺失时采用 RAFT-small fallback"),
    MethodSpec("V107算法（our network）", "v107", "内部单图模型", "v107 最新方法 100 epoch 长训练"),
]


def read_summary(path: Path) -> Dict[str, object] | None:
    """读取 summary，缺失时返回 None。"""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(summary: Dict[str, object], key: str) -> str:
    """按固定小数位格式化指标，保证 Word 回填稳定。"""

    value = float(summary[METRIC_KEYS[key]])
    return f"{value:.6f}"


def build_rows(out_root: Path) -> tuple[List[Dict[str, str]], List[str]]:
    """生成 CSV 行和状态说明。"""

    rows: List[Dict[str, str]] = []
    status: List[str] = []

    for dataset in DATASETS:
        for method in METHODS:
            summary_path = out_root / dataset.key / method.subdir / "eval" / "eval_summary.json"
            summary = read_summary(summary_path)
            if summary is None:
                status.append(f"- {dataset.label} / {method.name}: 缺少 `{summary_path}`，未完成。")
                continue
            rows.append(
                {
                    "dataset_key": dataset.key,
                    "dataset": dataset.label,
                    "method": method.name,
                    "method_type": method.method_type,
                    "num_samples": str(summary.get("num_samples", "")),
                    "crack_epe": fmt_metric(summary, "crack_epe"),
                    "global_epe": fmt_metric(summary, "global_epe"),
                    "dice": fmt_metric(summary, "dice"),
                    "crack_edge": fmt_metric(summary, "crack_edge"),
                    "global_edge": fmt_metric(summary, "global_edge"),
                    "folding": fmt_metric(summary, "folding"),
                    "summary_path": str(summary_path),
                    "note": method.note,
                }
            )
            status.append(
                f"- {dataset.label} / {method.name}: completed，样本数 {summary.get('num_samples', 'unknown')}。"
            )
    return rows, status


def write_csv(rows: Iterable[Dict[str, str]], path: Path) -> None:
    """写出机器可读 CSV。"""

    fieldnames = [
        "dataset_key",
        "dataset",
        "method",
        "method_type",
        "num_samples",
        "crack_epe",
        "global_epe",
        "dice",
        "crack_edge",
        "global_edge",
        "folding",
        "summary_path",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], status: List[str], path: Path) -> None:
    """写出便于人工检查的 Markdown 汇总。"""

    lines = [
        "# Roboflow 两个评估集外部四模型 + v107 指标",
        "",
        "说明：外部四模型为 oracle-pair dense matching 参考/上界，v107 为单图像校正模型，二者输入条件不同。",
        "",
        "| 数据集 | 方法 | 类型 | 样本数 | Crack EPE↓ | Global EPE↓ | Dice↑ | Crack Edge↑ | Global Edge↑ | Folding↓ | 说明 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {method_type} | {num_samples} | {crack_epe} | {global_epe} | {dice} | {crack_edge} | {global_edge} | {folding} | {note} |".format(
                **row
            )
        )
    lines.extend(["", "## 完成状态", "", *status, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Summarize Roboflow external/v107 evaluation")
    parser.add_argument(
        "--out-root",
        default="output_crackwarp_slurm/roboflow_external_v107_eval",
        help="Roboflow 补评估输出根目录",
    )
    return parser.parse_args()


def main() -> None:
    """主入口。"""

    args = parse_args()
    out_root = Path(args.out_root).resolve()
    rows, status = build_rows(out_root)
    write_csv(rows, out_root / "roboflow_external_v107_metrics.csv")
    write_markdown(rows, status, out_root / "roboflow_external_v107_metrics.md")
    print(f"summary saved to: {out_root}")
    print(f"completed rows: {len(rows)}")


if __name__ == "__main__":
    main()
