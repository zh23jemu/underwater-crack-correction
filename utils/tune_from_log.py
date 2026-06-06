#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Log-driven tuning helper.

1. Parse train.log epoch summaries.
2. Diagnose underfitting/overfitting/plateau patterns.
3. Export a compact hyperparameter sweep plan JSON.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config_crack as config


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train Loss:\s+([0-9eE\.\-]+)\s+EPE:\s+([0-9eE\.\-]+)px\s+\|\s+"
    r"Val Loss:\s+([0-9eE\.\-]+)\s+EPE:\s+([0-9eE\.\-]+)px\s+\|\s+LR:\s+([0-9eE\.\-]+)"
)


@dataclass
class EpochStat:
    epoch: int
    total_epochs: int
    train_loss: float
    train_epe: float
    val_loss: float
    val_epe: float
    lr: float


def parse_log(path: str) -> List[EpochStat]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            out.append(EpochStat(
                epoch=int(m.group(1)),
                total_epochs=int(m.group(2)),
                train_loss=float(m.group(3)),
                train_epe=float(m.group(4)),
                val_loss=float(m.group(5)),
                val_epe=float(m.group(6)),
                lr=float(m.group(7)),
            ))
    return out


def linear_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=np.float32)
    if len(y) < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum() + 1e-12
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def build_sweep(base: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    trials = []
    if mode == "underfit":
        factors = [1.4, 1.0, 0.8]
        for i, f in enumerate(factors):
            t = dict(base)
            t["name"] = f"underfit_lr_{i+1}"
            t["lr"] = round(base["lr"] * f, 8)
            t["w_photo"] = round(base["w_photo"] * (1.0 + 0.2 * i), 4)
            t["mixup_start_epoch"] = max(base["mixup_start_epoch"], 35)
            trials.append(t)
    elif mode == "overfit":
        variants = [(0.8, 0.20, 0.12), (0.7, 0.22, 0.15), (0.9, 0.18, 0.10)]
        for i, (lr_f, wd, smooth) in enumerate(variants):
            t = dict(base)
            t["name"] = f"overfit_reg_{i+1}"
            t["lr"] = round(base["lr"] * lr_f, 8)
            t["weight_decay"] = wd if wd > 1 else base["weight_decay"] * (1.0 + wd)
            t["w_smooth"] = smooth
            t["mixup_prob"] = min(0.2, base["mixup_prob"] + 0.03)
            trials.append(t)
    else:
        # plateau/oscillation
        variants = [
            ("plateau_lr_down", 0.7, base["w_photo"] + 0.05, base["w_ssim"] + 0.03),
            ("plateau_photo_up", 1.0, base["w_photo"] + 0.08, base["w_ssim"] + 0.05),
            ("plateau_smooth_up", 0.9, base["w_photo"], base["w_ssim"]),
        ]
        for name, lr_f, wp, ws in variants:
            t = dict(base)
            t["name"] = name
            t["lr"] = round(base["lr"] * lr_f, 8)
            t["w_photo"] = round(float(wp), 4)
            t["w_ssim"] = round(float(ws), 4)
            if name == "plateau_smooth_up":
                t["w_smooth"] = round(base["w_smooth"] * 1.25, 4)
            trials.append(t)
    return trials


def diagnose(stats: List[EpochStat], tail=10):
    val_epe = np.array([s.val_epe for s in stats], dtype=np.float32)
    train_epe = np.array([s.train_epe for s in stats], dtype=np.float32)
    val_loss = np.array([s.val_loss for s in stats], dtype=np.float32)
    train_loss = np.array([s.train_loss for s in stats], dtype=np.float32)

    best_idx = int(np.argmin(val_epe))
    best = stats[best_idx]
    last_n = min(tail, len(stats))
    slope_val_epe = linear_slope(val_epe[-last_n:])
    slope_train_epe = linear_slope(train_epe[-last_n:])
    gap_epe = float(np.mean(val_epe[-last_n:] - train_epe[-last_n:]))
    val_std = float(np.std(val_epe[-last_n:]))
    progress = float(val_epe[0] - val_epe[-1]) if len(val_epe) > 1 else 0.0

    if gap_epe > 25 and slope_train_epe < -0.2 and slope_val_epe > -0.05:
        mode = "overfit"
    elif progress < 15 and slope_train_epe > -0.05 and slope_val_epe > -0.05:
        mode = "underfit"
    elif val_std > 3.0:
        mode = "oscillation"
    else:
        mode = "plateau"

    report = {
        "num_epochs": len(stats),
        "best_epoch": best.epoch,
        "best_val_epe": best.val_epe,
        "best_val_loss": best.val_loss,
        "latest_val_epe": float(val_epe[-1]),
        "latest_train_epe": float(train_epe[-1]),
        "tail_slope_val_epe": slope_val_epe,
        "tail_slope_train_epe": slope_train_epe,
        "tail_gap_epe": gap_epe,
        "tail_std_val_epe": val_std,
        "diagnosis": mode,
        "tail_mean_val_loss": float(np.mean(val_loss[-last_n:])),
        "tail_mean_train_loss": float(np.mean(train_loss[-last_n:])),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description="Generate tuning recommendations from train.log")
    ap.add_argument("--log", default="output_crackwarp/train.log", help="path to train log")
    ap.add_argument("--out", default="output_crackwarp/tuning_plan.json", help="output json path")
    ap.add_argument("--tail", type=int, default=10, help="tail epochs for trend diagnosis")
    args = ap.parse_args()

    stats = parse_log(args.log)
    if not stats:
        raise RuntimeError(f"no epoch summary lines parsed from: {args.log}")

    report = diagnose(stats, tail=args.tail)
    base = {
        "lr": float(config.lr),
        "weight_decay": float(config.weight_decay),
        "w_coord": float(config.w_coord),
        "w_smooth": float(config.w_smooth),
        "w_fold": float(config.w_fold),
        "w_photo": float(config.w_photo),
        "w_ssim": float(config.w_ssim),
        "w_freq": float(config.w_freq),
        "mixup_start_epoch": int(config.mixup_start_epoch),
        "mixup_prob": float(config.mixup_prob),
        "mixup_alpha": float(config.mixup_alpha),
    }
    trials = build_sweep(base, report["diagnosis"])

    output = {
        "log_path": args.log,
        "diagnosis": report,
        "baseline": base,
        "suggested_trials": trials,
        "usage": [
            "Copy one suggested trial into config_crack.py",
            "Train for 15-25 epochs first (quick screening)",
            "Run utils/evaluate_metrics.py for objective comparison",
        ],
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=== Tuning Diagnosis ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"\nTuning plan written to: {args.out}")


if __name__ == "__main__":
    main()
