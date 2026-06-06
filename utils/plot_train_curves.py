#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


PATTERN = re.compile(
    r"Epoch (\d+)/(\d+) \| Train Loss: ([0-9.]+) EPE: ([0-9.]+)px \| "
    r"Val Loss: ([0-9.]+) EPE: ([0-9.]+)px \| LR: ([0-9.eE+-]+)"
)


def parse_log(log_path: Path):
    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = PATTERN.search(line)
        if not m:
            continue
        e, total, tr_loss, tr_epe, va_loss, va_epe, lr = m.groups()
        rows.append(
            {
                "epoch": int(e),
                "total": int(total),
                "train_loss": float(tr_loss),
                "train_epe": float(tr_epe),
                "val_loss": float(va_loss),
                "val_epe": float(va_epe),
                "lr": float(lr),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot train curves from train.log")
    parser.add_argument("--log", default="output_crackwarp/train.log", help="path to train.log")
    parser.add_argument("--out_dir", default="output_crackwarp", help="output directory")
    args = parser.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_log(log_path)
    if not rows:
        raise RuntimeError(f"No epoch summary lines found in {log_path}")

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_epe = [r["train_epe"] for r in rows]
    val_epe = [r["val_epe"] for r in rows]
    lrs = [r["lr"] for r in rows]

    best_val_loss_idx = min(range(len(rows)), key=lambda i: val_loss[i])
    best_val_epe_idx = min(range(len(rows)), key=lambda i: val_epe[i])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0, 0].scatter(
        [epochs[best_val_loss_idx]],
        [val_loss[best_val_loss_idx]],
        color="red",
        zorder=5,
        label=f"Best Val Loss E{epochs[best_val_loss_idx]}",
    )
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_epe, label="Train EPE(px)", linewidth=2)
    axes[0, 1].plot(epochs, val_epe, label="Val EPE(px)", linewidth=2)
    axes[0, 1].scatter(
        [epochs[best_val_epe_idx]],
        [val_epe[best_val_epe_idx]],
        color="red",
        zorder=5,
        label=f"Best Val EPE E{epochs[best_val_epe_idx]}",
    )
    axes[0, 1].set_title("EPE Curve")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("EPE (px)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, lrs, label="Learning Rate", color="tab:green", linewidth=2)
    axes[1, 0].set_title("LR Schedule")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    summary_text = (
        f"Epochs parsed: {len(rows)} / {rows[0]['total']}\n"
        f"Final Epoch: {epochs[-1]}\n"
        f"Final Val Loss: {val_loss[-1]:.5f}\n"
        f"Final Val EPE: {val_epe[-1]:.3f}px\n"
        f"Best Val Loss: {val_loss[best_val_loss_idx]:.5f} (E{epochs[best_val_loss_idx]})\n"
        f"Best Val EPE: {val_epe[best_val_epe_idx]:.3f}px (E{epochs[best_val_epe_idx]})"
    )
    axes[1, 1].text(0.02, 0.98, summary_text, va="top", fontsize=12, family="monospace")

    fig.suptitle("CrackWarp Training Curves", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_png = out_dir / "train_curves.png"
    out_svg = out_dir / "train_curves.svg"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_svg)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
