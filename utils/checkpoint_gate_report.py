#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint gate report for crack restoration quality.

What this script does:
1) Evaluate one or more checkpoints with utils/evaluate_metrics.py.
2) Judge each checkpoint against fixed quality gates.
3) Recommend the best checkpoint using crack-centric priority.
4) Export JSON/CSV/Markdown reports.

Usage example:
python utils/checkpoint_gate_report.py ^
  --output_dir output_crackwarp ^
  --img_dir underwater_crack_v3 ^
  --num 300 --batch_size 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Fixed thresholds agreed in this thread.
GATES = {
    "minimum": {
        "primary_crack_epe_px_mean": ("<=", 45.0),
        "primary_warp_crack_dice_mean": (">=", 0.35),
        "primary_crack_edge_fidelity_mean": (">=", 0.65),
        "global_epe_px_mean": ("<=", 40.0),
        "folding_rate_mean": ("<=", 0.05),
    },
    "good": {
        "primary_crack_epe_px_mean": ("<=", 30.0),
        "primary_warp_crack_dice_mean": (">=", 0.50),
        "primary_crack_edge_fidelity_mean": (">=", 0.75),
        "global_epe_px_mean": ("<=", 40.0),
        "folding_rate_mean": ("<=", 0.05),
    },
    "excellent": {
        "primary_crack_epe_px_mean": ("<=", 20.0),
        "primary_warp_crack_dice_mean": (">=", 0.65),
        "primary_crack_edge_fidelity_mean": (">=", 0.82),
        "global_epe_px_mean": ("<=", 40.0),
        "folding_rate_mean": ("<=", 0.05),
    },
}


@dataclass
class CheckpointResult:
    checkpoint: str
    summary: Dict[str, float]
    level: str
    passed: bool
    failed_rules: List[str]


def _check_rule(value: float, op: str, target: float) -> bool:
    if op == "<=":
        return value <= target
    if op == ">=":
        return value >= target
    raise ValueError(f"Unsupported op: {op}")


def _gate_status(summary: Dict[str, float], level_name: str) -> Tuple[bool, List[str]]:
    failed = []
    for key, (op, target) in GATES[level_name].items():
        value = float(summary.get(key, float("nan")))
        ok = _check_rule(value, op, target)
        if not ok:
            failed.append(f"{key} {op} {target} (actual={value:.6f})")
    return len(failed) == 0, failed


def classify_level(summary: Dict[str, float]) -> Tuple[str, bool, List[str]]:
    ok_exc, fail_exc = _gate_status(summary, "excellent")
    if ok_exc:
        return "excellent", True, []

    ok_good, fail_good = _gate_status(summary, "good")
    if ok_good:
        return "good", True, []

    ok_min, fail_min = _gate_status(summary, "minimum")
    if ok_min:
        return "minimum", True, []

    return "below_minimum", False, fail_min


def discover_checkpoints(output_dir: Path) -> List[Path]:
    preferred = [
        "best_crack_epe.pth",
        "best_epe.pth",
        "best_loss.pth",
        "final.pth",
    ]
    found: List[Path] = []
    for name in preferred:
        p = output_dir / name
        if p.exists():
            found.append(p)

    epoch_ckpts = sorted(output_dir.glob("epoch_*.pth"))
    found.extend([p for p in epoch_ckpts if p not in found])
    return found


def run_eval_for_checkpoint(
    repo_root: Path,
    ckpt: Path,
    img_dir: Path,
    eval_base_dir: Path,
    num: int,
    batch_size: int,
    size: int,
    gpu: int,
) -> Dict[str, float]:
    out_dir = eval_base_dir / ckpt.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "utils/evaluate_metrics.py",
        "--model",
        str(ckpt),
        "--img_dir",
        str(img_dir),
        "--out_dir",
        str(out_dir),
        "--size",
        str(size),
        "--batch_size",
        str(batch_size),
        "--num",
        str(num),
        "--gpu",
        str(gpu),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_train_log(train_log_path: Path) -> Dict[str, object]:
    if not train_log_path.exists():
        return {"exists": False}

    text = train_log_path.read_text(encoding="utf-8", errors="ignore")
    # Supports both old format and new format with CrackEPE.
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)\s+EPE:\s+([0-9.]+)px\s+\|"
        r"\s+Val Loss:\s+([0-9.]+)\s+EPE:\s+([0-9.]+)px(?:\s+CrackEPE:\s+([0-9.]+)px)?"
    )
    rows = []
    for m in pattern.finditer(text):
        rows.append(
            {
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "train_loss": float(m.group(3)),
                "train_epe": float(m.group(4)),
                "val_loss": float(m.group(5)),
                "val_epe": float(m.group(6)),
                "val_crack_epe": float(m.group(7)) if m.group(7) is not None else None,
            }
        )
    if not rows:
        return {"exists": True, "parsed_epochs": 0}
    return {
        "exists": True,
        "parsed_epochs": len(rows),
        "last_epoch": rows[-1],
        "best_val_epe": min(r["val_epe"] for r in rows),
        "best_val_loss": min(r["val_loss"] for r in rows),
        "best_val_crack_epe": min(
            (r["val_crack_epe"] for r in rows if r["val_crack_epe"] is not None),
            default=None,
        ),
    }


def rank_key(result: CheckpointResult) -> Tuple[float, float, float, float]:
    s = result.summary
    # Primary: lower crack EPE; secondary: higher crack dice and edge fidelity; then lower global EPE.
    return (
        float(s["primary_crack_epe_px_mean"]),
        -float(s["primary_warp_crack_dice_mean"]),
        -float(s["primary_crack_edge_fidelity_mean"]),
        float(s["global_epe_px_mean"]),
    )


def write_reports(report_dir: Path, results: List[CheckpointResult], train_log_info: Dict[str, object], best_ckpt: str):
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "gate_report.json"
    csv_path = report_dir / "gate_report.csv"
    md_path = report_dir / "gate_report.md"

    payload = {
        "best_checkpoint": best_ckpt,
        "train_log": train_log_info,
        "thresholds": GATES,
        "results": [
            {
                "checkpoint": r.checkpoint,
                "level": r.level,
                "passed": r.passed,
                "failed_rules": r.failed_rules,
                "summary": r.summary,
            }
            for r in results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "checkpoint",
        "level",
        "passed",
        "primary_crack_epe_px_mean",
        "primary_warp_crack_dice_mean",
        "primary_crack_edge_fidelity_mean",
        "global_epe_px_mean",
        "folding_rate_mean",
        "failed_rules",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            s = r.summary
            writer.writerow(
                {
                    "checkpoint": r.checkpoint,
                    "level": r.level,
                    "passed": r.passed,
                    "primary_crack_epe_px_mean": s.get("primary_crack_epe_px_mean"),
                    "primary_warp_crack_dice_mean": s.get("primary_warp_crack_dice_mean"),
                    "primary_crack_edge_fidelity_mean": s.get("primary_crack_edge_fidelity_mean"),
                    "global_epe_px_mean": s.get("global_epe_px_mean"),
                    "folding_rate_mean": s.get("folding_rate_mean"),
                    "failed_rules": " | ".join(r.failed_rules),
                }
            )

    lines = []
    lines.append("# Crack Restoration Gate Report")
    lines.append("")
    lines.append(f"- Best checkpoint (crack-priority ranking): `{best_ckpt}`")
    lines.append("")
    lines.append("## Gate Summary")
    lines.append("")
    for r in results:
        s = r.summary
        lines.append(
            f"- `{r.checkpoint}` | level={r.level} | pass={r.passed} | "
            f"crack_epe={s.get('primary_crack_epe_px_mean', float('nan')):.3f} | "
            f"dice={s.get('primary_warp_crack_dice_mean', float('nan')):.3f} | "
            f"edge={s.get('primary_crack_edge_fidelity_mean', float('nan')):.3f} | "
            f"global_epe={s.get('global_epe_px_mean', float('nan')):.3f} | "
            f"folding={s.get('folding_rate_mean', float('nan')):.3f}"
        )
        if r.failed_rules:
            lines.append(f"  failed: {'; '.join(r.failed_rules)}")
    lines.append("")
    lines.append("## Train Log Parse")
    lines.append("")
    lines.append(f"```json\n{json.dumps(train_log_info, ensure_ascii=False, indent=2)}\n```")

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Generate crack-restoration gate report for checkpoints.")
    p.add_argument("--output_dir", default="output_crackwarp", help="Directory containing checkpoints and train.log")
    p.add_argument("--img_dir", default="underwater_crack_v3", help="Dataset directory for evaluation")
    p.add_argument("--num", type=int, default=300, help="Number of samples for evaluation, -1 for all")
    p.add_argument("--batch_size", type=int, default=4, help="Eval batch size")
    p.add_argument("--size", type=int, default=512, help="Eval image size")
    p.add_argument("--gpu", type=int, default=0, help="GPU id")
    p.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional explicit checkpoint paths. If empty, auto-discover in output_dir.",
    )
    p.add_argument(
        "--report_dir",
        default=None,
        help="Where to write gate_report.*. Defaults to <output_dir>/gate_report",
    )
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = (repo_root / args.output_dir).resolve()
    img_dir = (repo_root / args.img_dir).resolve()

    if args.checkpoints:
        ckpts = [Path(c).resolve() for c in args.checkpoints]
    else:
        ckpts = discover_checkpoints(output_dir)

    if not ckpts:
        raise RuntimeError(f"No checkpoints found under: {output_dir}")

    report_dir = Path(args.report_dir).resolve() if args.report_dir else (output_dir / "gate_report")
    eval_base_dir = report_dir / "eval_cache"
    eval_base_dir.mkdir(parents=True, exist_ok=True)

    results: List[CheckpointResult] = []
    for ckpt in ckpts:
        print(f"[Gate] Evaluating: {ckpt}")
        summary = run_eval_for_checkpoint(
            repo_root=repo_root,
            ckpt=ckpt,
            img_dir=img_dir,
            eval_base_dir=eval_base_dir,
            num=args.num,
            batch_size=args.batch_size,
            size=args.size,
            gpu=args.gpu,
        )
        level, passed, failed = classify_level(summary)
        results.append(
            CheckpointResult(
                checkpoint=str(ckpt),
                summary=summary,
                level=level,
                passed=passed,
                failed_rules=failed,
            )
        )

    results.sort(key=rank_key)
    best_ckpt = results[0].checkpoint

    train_log_info = parse_train_log(output_dir / "train.log")
    write_reports(report_dir, results, train_log_info, best_ckpt)

    print("\n[Gate] Done.")
    print(f"[Gate] Best checkpoint: {best_ckpt}")
    print(f"[Gate] Report dir: {report_dir}")
    if not any(r.passed for r in results):
        print("[Gate] No checkpoint reached minimum gate.")


if __name__ == "__main__":
    main()

