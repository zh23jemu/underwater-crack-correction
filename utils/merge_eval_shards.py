#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并 `evaluate_metrics.py` 的分片评估结果。

典型场景：
1. 在 Slurm 限时分区上用 `--num_shards/--shard_index` 把全量评估拆成多个任务。
2. 每个任务输出一个 `eval_per_image.csv`。
3. 本脚本读取所有分片 CSV，去重后重新计算整体均值/标准差，并输出统一的
   `eval_summary.json` 和 `eval_per_image.csv`。

注意：本脚本只合并已经算好的逐图指标，不重新跑模型，因此速度很快。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


FIELDNAMES = [
    'image',
    'crack_epe_px',
    'crack_edge_fidelity',
    'warp_crack_dice',
    'global_epe_px',
    'global_edge_fidelity',
    'folding_rate',
    'crack_ratio',
]


def read_rows(csv_paths: Iterable[Path]) -> List[Dict[str, str]]:
    """读取多个分片 CSV，并按图片名去重，防止误重复合并。"""

    merged: Dict[str, Dict[str, str]] = {}
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f'缺少分片 CSV: {csv_path}')
        with csv_path.open('r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            missing = [name for name in FIELDNAMES if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f'{csv_path} 缺少列: {missing}')
            for row in reader:
                image = row['image']
                if image in merged:
                    raise ValueError(f'图片重复出现在多个分片中: {image}')
                merged[image] = {name: row[name] for name in FIELDNAMES}

    return [merged[name] for name in sorted(merged)]


def values(rows: List[Dict[str, str]], key: str) -> np.ndarray:
    """从逐图行中提取某个数值列。"""

    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def build_summary(args: argparse.Namespace, rows: List[Dict[str, str]]) -> Dict[str, object]:
    """按 `evaluate_metrics.py` 的字段口径重新计算整体 summary。"""

    crack_epe = values(rows, 'crack_epe_px')
    crack_edge = values(rows, 'crack_edge_fidelity')
    dice = values(rows, 'warp_crack_dice')
    global_epe = values(rows, 'global_epe_px')
    global_edge = values(rows, 'global_edge_fidelity')
    folding = values(rows, 'folding_rate')
    crack_ratio = values(rows, 'crack_ratio')

    return {
        'model': args.model,
        'img_dir': args.img_dir,
        'num_samples': int(len(rows)),
        'primary_crack_epe_px_mean': float(np.mean(crack_epe)),
        'primary_crack_epe_px_std': float(np.std(crack_epe)),
        'primary_crack_edge_fidelity_mean': float(np.mean(crack_edge)),
        'primary_crack_edge_fidelity_std': float(np.std(crack_edge)),
        'primary_warp_crack_dice_mean': float(np.mean(dice)),
        'primary_warp_crack_dice_std': float(np.std(dice)),
        'global_epe_px_mean': float(np.mean(global_epe)),
        'global_epe_px_std': float(np.std(global_epe)),
        'global_edge_fidelity_mean': float(np.mean(global_edge)),
        'global_edge_fidelity_std': float(np.std(global_edge)),
        'folding_rate_mean': float(np.mean(folding)),
        'folding_rate_std': float(np.std(folding)),
        'crack_ratio_mean': float(np.mean(crack_ratio)),
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description='Merge sharded CrackWarp evaluation outputs')
    parser.add_argument('--shard_dirs', nargs='+', required=True, help='directories containing eval_per_image.csv')
    parser.add_argument('--out_dir', required=True, help='merged output directory')
    parser.add_argument('--model', required=True, help='model path recorded in merged summary')
    parser.add_argument('--img_dir', required=True, help='image directory recorded in merged summary')
    return parser.parse_args()


def main() -> None:
    """合并分片结果并写出 summary / per-image CSV。"""

    args = parse_args()
    shard_csvs = [Path(d) / 'eval_per_image.csv' for d in args.shard_dirs]
    rows = read_rows(shard_csvs)
    if not rows:
        raise RuntimeError('没有读到任何逐图评估结果')

    summary = build_summary(args, rows)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = Path(args.out_dir) / 'eval_summary.json'
    per_image_path = Path(args.out_dir) / 'eval_per_image.csv'

    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with per_image_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print('=== Merged Evaluation Summary ===')
    for key, value in summary.items():
        if isinstance(value, float):
            print(f'{key}: {value:.6f}')
        else:
            print(f'{key}: {value}')
    print(f'summary saved to: {summary_path}')
    print(f'per-image saved to: {per_image_path}')


if __name__ == '__main__':
    main()
