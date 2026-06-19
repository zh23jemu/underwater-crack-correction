#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从逐图对比结果中生成 hard-sample 训练清单。

用途：
  当前 v8/v9 常规微调已经进入收益变小阶段，剩余短板主要来自
  v8 相对旧模型仍失败或共同失败的样本。这个脚本把 compare 目录中的
  `old_better_images.txt`、`joint_failure_images.txt` 等诊断清单汇总成
  一个训练可用的 hard-sample 列表。

输出模式：
  - family：输出原始裂缝族群编号，例如 crack0563。训练时会匹配该族群
    的所有合成样本，适合“某类裂缝整体困难”的场景。
  - image：输出完整图片名，例如 crack0563_01.png，只强化具体样本。
  - both：同时输出族群编号和完整图片名。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_image_names(compare_dirs: Iterable[Path], list_files: Iterable[str]) -> Iterable[str]:
    """遍历多个 compare 目录，读取指定诊断清单中的图片名。"""
    for compare_dir in compare_dirs:
        for name in list_files:
            path = compare_dir / name
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    image = line.strip()
                    if image and not image.startswith("#"):
                        yield image


def image_family(image_name: str) -> str:
    """把 crack0563_01.png 归并为 crack0563 族群编号。"""
    stem = Path(image_name).stem
    return stem.rsplit("_", 1)[0]


def build_items(images: list[str], mode: str) -> Counter[str]:
    """按输出模式构建带频次的 hard-sample 条目。"""
    counter: Counter[str] = Counter()
    for image in images:
        if mode in {"image", "both"}:
            counter[image] += 1
        if mode in {"family", "both"}:
            counter[image_family(image)] += 1
    return counter


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Build hard-sample list from compare outputs")
    parser.add_argument(
        "--compare_dirs",
        nargs="+",
        required=True,
        help="一个或多个 compare_eval_per_image.py 输出目录。",
    )
    parser.add_argument("--out", required=True, help="输出 hard-sample 清单路径。")
    parser.add_argument(
        "--list_files",
        nargs="+",
        default=["old_better_images.txt", "joint_failure_images.txt"],
        help="从 compare 目录中读取哪些图片清单。",
    )
    parser.add_argument(
        "--mode",
        choices=["family", "image", "both"],
        default="family",
        help="输出原图族群、具体图片，或两者都输出。",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="条目至少出现多少次才写入，用于过滤偶然失败样本。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="只保留频次最高的前 K 个条目；0 表示不截断。",
    )
    return parser.parse_args()


def main() -> None:
    """生成 hard-sample 清单并打印摘要，方便直接复制到 Slurm 训练。"""
    args = parse_args()
    compare_dirs = [Path(p) for p in args.compare_dirs]
    images = list(iter_image_names(compare_dirs, args.list_files))
    counter = build_items(images, args.mode)

    items = [(key, count) for key, count in counter.most_common() if count >= args.min_count]
    if args.topk > 0:
        items = items[: args.topk]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for key, _ in items:
            f.write(f"{key}\n")

    print("hard_sample_list:", out_path)
    print("source_images:", len(images))
    print("written_items:", len(items))
    print("top_items:")
    for key, count in items[:20]:
        print(f"{key}\t{count}")


if __name__ == "__main__":
    main()
