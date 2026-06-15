#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crack-centric evaluation for CrackWarp models.

Primary metrics:
1) crack_epe_px: flow endpoint error in crack ROI.
2) crack_edge_fidelity: edge consistency in crack ROI after warping.
3) warp_crack_dice: Dice overlap between pred-warped and GT-warped crack masks.

Reference metrics:
- global_epe_px
- global_edge_fidelity
- folding_rate
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.crack_warp_net import build_crack_warp_net
from loss_crack import CrackMaskEstimator


def load_model(model_path: str, device: torch.device):
    net = build_crack_warp_net().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt:
            state = ckpt['ema_state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
    else:
        state = ckpt
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def list_images(img_dir: str):
    out = []
    for f in sorted(os.listdir(img_dir)):
        low = f.lower()
        if low.endswith(('.png', '.jpg', '.jpeg')):
            out.append(os.path.join(img_dir, f))
    return out


def load_image_tensor(path: str, size=(512, 512)):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f'failed to read image: {path}')
    img_bgr = cv2.resize(img_bgr, size)
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def load_label_tensor(path: str, size=(512, 512)):
    label = np.load(path).astype(np.float32)
    if label.shape != (2, size[1], size[0]):
        label = np.stack([
            cv2.resize(label[0], size, interpolation=cv2.INTER_LINEAR),
            cv2.resize(label[1], size, interpolation=cv2.INTER_LINEAR),
        ], axis=0).astype(np.float32)
    label = np.clip(label, 0.0, 1.0)
    return torch.from_numpy(label).float()


def flow_to_grid(flow):
    gx = flow[:, 0] * 2.0 - 1.0
    gy = flow[:, 1] * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1)


def flow_epe_px(pred, gt):
    diff = (pred - gt) * 511.0
    epe = torch.sqrt((diff ** 2).sum(dim=1))
    return epe.mean(dim=(1, 2))


def flow_epe_px_masked(pred, gt, mask):
    diff = (pred - gt) * 511.0
    epe = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True))
    num = (epe * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return num / den


def folding_rate(flow):
    u = flow[:, 0]
    v = flow[:, 1]
    du_dx = (u[:, :, 2:] - u[:, :, :-2]) * 0.5
    du_dy = (u[:, 2:, :] - u[:, :-2, :]) * 0.5
    dv_dx = (v[:, :, 2:] - v[:, :, :-2]) * 0.5
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) * 0.5

    du_dx = du_dx[:, 1:-1, :]
    dv_dy = dv_dy[:, :, 1:-1]
    du_dy = du_dy[:, :, 1:-1]
    dv_dx = dv_dx[:, 1:-1, :]
    det = du_dx * dv_dy - du_dy * dv_dx
    return (det <= 0).float().mean(dim=(1, 2))


def sobel_mag(gray):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def edge_fidelity(pred_flow, gt_flow, img, mask=None):
    pred_img = F.grid_sample(img, flow_to_grid(pred_flow), mode='bilinear', padding_mode='border', align_corners=True)
    gt_img = F.grid_sample(img, flow_to_grid(gt_flow), mode='bilinear', padding_mode='border', align_corners=True)
    pred_gray = pred_img.mean(dim=1, keepdim=True)
    gt_gray = gt_img.mean(dim=1, keepdim=True)
    pred_g = sobel_mag(pred_gray)
    gt_g = sobel_mag(gt_gray)

    if mask is None:
        num = (pred_g - gt_g).abs().mean(dim=(1, 2, 3))
        den = gt_g.abs().mean(dim=(1, 2, 3)) + 1e-6
    else:
        num = ((pred_g - gt_g).abs() * mask).sum(dim=(1, 2, 3))
        den = (gt_g.abs() * mask).sum(dim=(1, 2, 3)) + 1e-6
    score = 1.0 - num / den
    return score.clamp(0.0, 1.0)


def detect_crack_mask_bgr(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # adaptive threshold: keep strongest thin dark responses
    thr = np.percentile(blackhat, 92.0)
    thr = max(float(thr), 8.0)
    m1 = (blackhat >= thr).astype(np.uint8) * 255

    # intensity prior: dark structures
    m2 = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)[1]

    mask = cv2.bitwise_and(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return (mask > 0).astype(np.float32)


def build_crack_mask_batch(imgs_rgb: torch.Tensor, masker: CrackMaskEstimator):
    """
    Build crack ROI mask from the same estimator used during training loss,
    so train/eval focus regions stay consistent.
    """
    out = masker(imgs_rgb).detach()
    # Avoid accidental all-zero mask, but never fallback to all-ones.
    # If too small, softly dilate existing response.
    small = (out.sum(dim=(1, 2, 3), keepdim=True) < 32.0)
    if small.any():
        dilated = F.max_pool2d(out, kernel_size=7, stride=1, padding=3)
        out = torch.where(small, dilated.clamp(0.0, 1.0), out)
    return out


def warp_tensor_img_to_bgr(img_rgb: torch.Tensor, flow: torch.Tensor):
    # img_rgb: (1,3,H,W), flow: (1,2,H,W)
    warped = F.grid_sample(img_rgb, flow_to_grid(flow), mode='bilinear', padding_mode='border', align_corners=True)
    np_img = warped[0].detach().cpu().numpy().transpose(1, 2, 0)
    return (np_img[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)


def warp_crack_dice(pred_flow, gt_flow, img_rgb, crack_masker: CrackMaskEstimator):
    b = pred_flow.shape[0]
    dices = []
    for i in range(b):
        in_img = img_rgb[i:i + 1]
        pred_warp = F.grid_sample(
            in_img,
            flow_to_grid(pred_flow[i:i + 1]),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        gt_warp = F.grid_sample(
            in_img,
            flow_to_grid(gt_flow[i:i + 1]),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        mp = crack_masker(pred_warp).detach()
        mg = crack_masker(gt_warp).detach()
        inter = float((mp * mg).sum().item())
        denom = float((mp.sum() + mg.sum()).item()) + 1e-6
        dices.append((2.0 * inter) / denom)
    return torch.tensor(dices, device=pred_flow.device)


@torch.no_grad()
def run_eval(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    net = load_model(args.model, device)
    crack_masker = CrackMaskEstimator(
        topk=args.crack_topk,
        temperature=args.crack_temp,
    ).to(device)
    crack_masker.eval()

    paths = list_images(args.img_dir)
    rows = []
    for p in paths:
        npy = p + '.npy'
        if not os.path.exists(npy):
            continue
        rows.append((p, npy))

    # 支持把全量评估拆成多个互不重叠的分片，适合在 gpuHz 这类限时分区上跑。
    # 推荐用法：同一个模型提交 N 个任务，分别设置 --num_shards N 和
    # --shard_index 0..N-1；某个分片失败时只需要重跑该分片。
    if args.num_shards < 1:
        raise ValueError('--num_shards 必须 >= 1')
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('--shard_index 必须满足 0 <= shard_index < num_shards')
    if args.start < 0:
        raise ValueError('--start 必须 >= 0')
    if args.start:
        rows = rows[args.start:]
    if args.num_shards > 1:
        rows = rows[args.shard_index::args.num_shards]
    if args.num > 0:
        rows = rows[:args.num]
    if not rows:
        raise RuntimeError(f'No paired image/label found under {args.img_dir}')

    global_epe_all = []
    crack_epe_all = []
    global_edge_all = []
    crack_edge_all = []
    fold_all = []
    warp_dice_all = []
    crack_ratio_all = []
    per_image = []

    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        imgs = torch.stack([load_image_tensor(x[0], size=(args.size, args.size)) for x in chunk], dim=0).to(device)
        gts = torch.stack([load_label_tensor(x[1], size=(args.size, args.size)) for x in chunk], dim=0).to(device)

        pred = net(imgs)
        if isinstance(pred, list):
            pred = pred[-1]

        crack_mask = build_crack_mask_batch(imgs, crack_masker)

        global_epe = flow_epe_px(pred, gts)
        crack_epe = flow_epe_px_masked(pred, gts, crack_mask)

        global_edge = edge_fidelity(pred, gts, imgs, mask=None)
        crack_edge = edge_fidelity(pred, gts, imgs, mask=crack_mask)

        fold = folding_rate(pred)
        warp_dice = warp_crack_dice(pred, gts, imgs, crack_masker=crack_masker)
        crack_ratio = crack_mask.mean(dim=(1, 2, 3))

        global_epe_all.append(global_epe.cpu().numpy())
        crack_epe_all.append(crack_epe.cpu().numpy())
        global_edge_all.append(global_edge.cpu().numpy())
        crack_edge_all.append(crack_edge.cpu().numpy())
        fold_all.append(fold.cpu().numpy())
        warp_dice_all.append(warp_dice.cpu().numpy())
        crack_ratio_all.append(crack_ratio.cpu().numpy())

        for j, (img_path, _) in enumerate(chunk):
            per_image.append({
                'image': str(Path(img_path).name),
                'crack_epe_px': float(crack_epe[j].item()),
                'crack_edge_fidelity': float(crack_edge[j].item()),
                'warp_crack_dice': float(warp_dice[j].item()),
                'global_epe_px': float(global_epe[j].item()),
                'global_edge_fidelity': float(global_edge[j].item()),
                'folding_rate': float(fold[j].item()),
                'crack_ratio': float(crack_ratio[j].item()),
            })

    global_epe_np = np.concatenate(global_epe_all, axis=0)
    crack_epe_np = np.concatenate(crack_epe_all, axis=0)
    global_edge_np = np.concatenate(global_edge_all, axis=0)
    crack_edge_np = np.concatenate(crack_edge_all, axis=0)
    fold_np = np.concatenate(fold_all, axis=0)
    warp_dice_np = np.concatenate(warp_dice_all, axis=0)
    crack_ratio_np = np.concatenate(crack_ratio_all, axis=0)

    summary = {
        'model': args.model,
        'img_dir': args.img_dir,
        'start': int(args.start),
        'num_shards': int(args.num_shards),
        'shard_index': int(args.shard_index),
        'num_samples': int(len(per_image)),
        'primary_crack_epe_px_mean': float(np.mean(crack_epe_np)),
        'primary_crack_epe_px_std': float(np.std(crack_epe_np)),
        'primary_crack_edge_fidelity_mean': float(np.mean(crack_edge_np)),
        'primary_crack_edge_fidelity_std': float(np.std(crack_edge_np)),
        'primary_warp_crack_dice_mean': float(np.mean(warp_dice_np)),
        'primary_warp_crack_dice_std': float(np.std(warp_dice_np)),
        'global_epe_px_mean': float(np.mean(global_epe_np)),
        'global_epe_px_std': float(np.std(global_epe_np)),
        'global_edge_fidelity_mean': float(np.mean(global_edge_np)),
        'global_edge_fidelity_std': float(np.std(global_edge_np)),
        'folding_rate_mean': float(np.mean(fold_np)),
        'folding_rate_std': float(np.std(fold_np)),
        'crack_ratio_mean': float(np.mean(crack_ratio_np)),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, 'eval_summary.json')
    per_image_path = os.path.join(args.out_dir, 'eval_per_image.csv')

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(per_image_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'image',
                'crack_epe_px',
                'crack_edge_fidelity',
                'warp_crack_dice',
                'global_epe_px',
                'global_edge_fidelity',
                'folding_rate',
                'crack_ratio',
            ],
        )
        writer.writeheader()
        writer.writerows(per_image)

    print('=== Evaluation Summary ===')
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'{k}: {v:.6f}')
        else:
            print(f'{k}: {v}')
    print(f'summary saved to: {summary_path}')
    print(f'per-image saved to: {per_image_path}')


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate CrackWarp model with crack-centric metrics')
    p.add_argument('--model', required=True, help='model path, e.g. output_crackwarp/best_epe.pth')
    p.add_argument('--img_dir', required=True, help='dataset directory containing images and paired .npy labels')
    p.add_argument('--out_dir', default='output_crackwarp/eval_metrics', help='output directory for metrics')
    p.add_argument('--size', type=int, default=512, help='evaluation image size')
    p.add_argument('--batch_size', type=int, default=4, help='batch size')
    p.add_argument('--num', type=int, default=-1, help='number of samples, -1 for all')
    p.add_argument('--start', type=int, default=0, help='skip the first N paired samples before evaluation')
    p.add_argument('--num_shards', type=int, default=1, help='split paired samples into N interleaved shards')
    p.add_argument('--shard_index', type=int, default=0, help='current shard index in [0, num_shards)')
    p.add_argument('--gpu', type=int, default=0, help='gpu index')
    p.add_argument('--crack_topk', type=float, default=0.08, help='top-k ratio for crack ROI estimator')
    p.add_argument('--crack_temp', type=float, default=0.07, help='temperature for crack ROI estimator')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(args)
