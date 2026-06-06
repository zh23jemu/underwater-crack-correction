#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export crack-focused local ROI comparisons for restoration quality inspection.

For each image:
1) Run model inference -> corrected image
2) Build crack mask using CrackMaskEstimator (same family as training loss)
3) Find top-K crack ROIs from connected components
4) Export local panels: [Input ROI | Corrected ROI | GT Corrected ROI(optional)]
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.crack_warp_net import build_crack_warp_net
from loss_crack import CrackMaskEstimator


def load_model(model_path: str, device: torch.device):
    net = build_crack_warp_net().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        if "ema_state_dict" in ckpt:
            state = ckpt["ema_state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def list_images(img_dir: str):
    out = []
    for f in sorted(os.listdir(img_dir)):
        low = f.lower()
        if low.endswith((".png", ".jpg", ".jpeg")) and not low.endswith(".npy"):
            out.append(os.path.join(img_dir, f))
    return out


def img_to_tensor(img_bgr: np.ndarray):
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)


def flow_to_grid(flow):
    gx = flow[:, 0] * 2.0 - 1.0
    gy = flow[:, 1] * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1)


def run_correct(net, img_bgr: np.ndarray, device: torch.device, size=512):
    img_512 = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    t = img_to_tensor(img_512).to(device)
    with torch.no_grad():
        flow = net(t)
        if isinstance(flow, list):
            flow = flow[-1]
    grid = flow_to_grid(flow)
    corrected = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    corrected_np = corrected[0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
    corrected_np = np.clip(corrected_np * 255.0, 0, 255).astype(np.uint8)
    return img_512, corrected_np, flow, t


def gt_correct(img_t: torch.Tensor, label_path: str, size=512):
    if not os.path.exists(label_path):
        return None
    gt = np.load(label_path).astype(np.float32)
    if gt.shape != (2, size, size):
        gt = np.stack([
            cv2.resize(gt[0], (size, size), interpolation=cv2.INTER_LINEAR),
            cv2.resize(gt[1], (size, size), interpolation=cv2.INTER_LINEAR),
        ], axis=0).astype(np.float32)
    gt_t = torch.from_numpy(gt).unsqueeze(0).to(img_t.device)
    warped = F.grid_sample(img_t, flow_to_grid(gt_t), mode="bilinear", padding_mode="border", align_corners=True)
    out = warped[0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def extract_rois(mask01: np.ndarray, max_rois=3, min_area=64, margin=16):
    m = (mask01 >= 0.5).astype(np.uint8) * 255
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comps.append((area, x, y, w, h))
    comps.sort(key=lambda z: z[0], reverse=True)
    rois = []
    H, W = mask01.shape
    for _, x, y, w, h in comps[:max_rois]:
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(W, x + w + margin)
        y1 = min(H, y + h + margin)
        rois.append((x0, y0, x1, y1))
    return rois


def panel_for_roi(inp, pred, gt, roi, name):
    x0, y0, x1, y1 = roi
    a = inp[y0:y1, x0:x1]
    b = pred[y0:y1, x0:x1]
    tiles = [a, b]
    labels = ["Input ROI", "Corrected ROI"]
    if gt is not None:
        c = gt[y0:y1, x0:x1]
        tiles.append(c)
        labels.append("GT Corrected ROI")

    h = max(t.shape[0] for t in tiles)
    resized = []
    for t in tiles:
        if t.shape[0] != h:
            scale = h / max(1, t.shape[0])
            w = int(round(t.shape[1] * scale))
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
        resized.append(t)
    row = np.hstack(resized)

    top = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
    cv2.putText(top, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
    out = np.vstack([top, row])
    return out


def main():
    p = argparse.ArgumentParser(description="Export crack ROI local comparisons")
    p.add_argument("--model", required=True, help="checkpoint path")
    p.add_argument("--img_dir", required=True, help="image directory")
    p.add_argument("--out_dir", default="output_crackwarp/crack_roi_visuals", help="output directory")
    p.add_argument("--num", type=int, default=20, help="number of images, -1 for all")
    p.add_argument("--size", type=int, default=512, help="inference size")
    p.add_argument("--gpu", type=int, default=0, help="gpu id")
    p.add_argument("--max_rois", type=int, default=3, help="max rois per image")
    p.add_argument("--min_area", type=int, default=64, help="min crack component area")
    p.add_argument("--margin", type=int, default=16, help="roi padding pixels")
    p.add_argument("--crack_topk", type=float, default=0.08, help="crack estimator topk")
    p.add_argument("--crack_temp", type=float, default=0.07, help="crack estimator temperature")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    net = load_model(args.model, device)
    masker = CrackMaskEstimator(topk=args.crack_topk, temperature=args.crack_temp).to(device).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = list_images(args.img_dir)
    if args.num > 0:
        imgs = imgs[:args.num]

    total_panels = 0
    for idx, img_path in enumerate(imgs, start=1):
        img = cv2.imread(img_path)
        if img is None:
            continue
        inp, pred, flow, img_t = run_correct(net, img, device, size=args.size)
        with torch.no_grad():
            mask = masker(img_t)[0, 0].detach().cpu().numpy()
        rois = extract_rois(mask, max_rois=args.max_rois, min_area=args.min_area, margin=args.margin)
        if not rois:
            continue

        gt = gt_correct(img_t, img_path + ".npy", size=args.size)
        stem = Path(img_path).stem
        vis = inp.copy()
        for rid, (x0, y0, x1, y1) in enumerate(rois, start=1):
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 220), 1)
            cv2.putText(vis, f"R{rid}", (x0 + 2, max(12, y0 + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{stem}_roi_boxes.png"), vis)

        for rid, roi in enumerate(rois, start=1):
            panel = panel_for_roi(inp, pred, gt, roi, f"{stem} | ROI-{rid} ({roi[0]},{roi[1]})-({roi[2]},{roi[3]})")
            cv2.imwrite(str(out_dir / f"{stem}_roi{rid:02d}_panel.png"), panel)
            total_panels += 1

        print(f"[{idx}/{len(imgs)}] {stem}: {len(rois)} ROI panels")

    print(f"Done. Panels exported: {total_panels}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

