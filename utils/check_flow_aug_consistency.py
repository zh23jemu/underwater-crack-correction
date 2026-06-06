#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sanity-check geometric augmentation consistency for flow labels.

The test uses identity flow label:
  flow_x[i,j] = j/(W-1), flow_y[i,j] = i/(H-1)
For a purely geometric augmentation, transformed image coordinates and
transformed flow should remain identity mapping at the new pixel locations.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MyDataSet import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90


def build_identity_flow(h: int, w: int) -> np.ndarray:
    x = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    y = np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1), (1, w))
    return np.stack([x, y], axis=0)


def max_abs_err_to_identity(flow: np.ndarray) -> float:
    h, w = flow.shape[1], flow.shape[2]
    ref = build_identity_flow(h, w)
    return float(np.max(np.abs(flow - ref)))


def main():
    h, w = 65, 97
    dummy_img = np.zeros((h, w, 3), dtype=np.float32)
    label = build_identity_flow(h, w)

    # Force each augmentation branch via p=1.
    tests = [
        ("hflip", RandomHorizontalFlip(p=1.0)),
        ("vflip", RandomVerticalFlip(p=1.0)),
        ("rot90_rand", RandomRotate90(p=1.0)),
    ]

    # Repeat rotate multiple times to cover k=1/2/3 random branches.
    errs = {}
    for name, aug in tests:
        max_err = 0.0
        rounds = 40 if name == "rot90_rand" else 1
        for _ in range(rounds):
            _, out = aug(dummy_img.copy(), label.copy())
            max_err = max(max_err, max_abs_err_to_identity(out))
        errs[name] = max_err

    print("=== Flow Aug Consistency ===")
    ok = True
    for k, v in errs.items():
        print(f"{k}: max_abs_err={v:.8f}")
        if v > 1e-5:
            ok = False

    if ok:
        print("PASS: geometric flow augmentations are consistent with identity mapping.")
    else:
        raise SystemExit("FAIL: found inconsistent geometric flow augmentation.")


if __name__ == "__main__":
    main()
