#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
epoch_80 模型推理脚本
用途：加载 output_crackwarp/epoch_80.pth，对数据集图像进行批量推理，
      生成矫正结果对比图（输入 | 矫正输出 | flow可视化）

运行：
    cd /root/autodl-tmp/GOOD_cnn
    python infer_epoch80.py
    # 或指定更多参数：
    python infer_epoch80.py --model output_crackwarp/epoch_80.pth \
                             --img_dir underwater_crack_v3 \
                             --out_dir output_crackwarp/epoch80_results \
                             --num 20
"""

import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from models.crack_warp_net import build_crack_warp_net


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """
    将 (2, H, W) flow [0,1] 转换为 HSV 彩色可视化图。
    色调 = 方向，饱和度 = 幅度。
    """
    fx = flow[0] - 0.5   # 中心化到 [-0.5, 0.5]
    fy = flow[1] - 0.5
    angle  = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)   # [0, 1]
    mag    = np.sqrt(fx ** 2 + fy ** 2)
    mag    = np.clip(mag / (mag.max() + 1e-6), 0, 1)

    hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 179).astype(np.uint8)   # Hue
    hsv[..., 1] = (mag   * 255).astype(np.uint8)   # Saturation
    hsv[..., 2] = 200                               # Value (固定亮度)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def compute_epe(flow_pred: np.ndarray, flow_gt: np.ndarray) -> float:
    """计算端点误差 EPE（像素单位，flow 值域 [0,1] × 512）"""
    diff = (flow_pred - flow_gt) * 512.0
    return float(np.sqrt((diff ** 2).sum(axis=0)).mean())


def load_model(model_path: str, device: torch.device):
    """
    加载模型权重。
    checkpoint 结构：
      - 训练保存的完整 checkpoint 含 'ema_state_dict' / 'model_state_dict' 等键
      - 也兼容直接是 state_dict 的情况
    优先使用 ema_state_dict（推理效果更好），其次 model_state_dict，最后直接当 state_dict。
    """
    net = build_crack_warp_net().to(device)
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt:
            state = ckpt['ema_state_dict']
            print('[INFO] Using ema_state_dict from checkpoint')
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
            print('[INFO] Using model_state_dict from checkpoint')
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
            print('[INFO] Using state_dict from checkpoint')
        else:
            state = ckpt
            print('[INFO] Using checkpoint directly as state_dict')
    else:
        state = ckpt
        print('[INFO] Checkpoint is raw state_dict')

    # 去掉 'module.' 前缀（DataParallel）
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def preprocess(img_bgr: np.ndarray, size=(512, 512)) -> torch.Tensor:
    """BGR uint8 → RGB float tensor (1,3,H,W)"""
    img = cv2.resize(img_bgr, size).astype(np.float32) / 255.0
    # Channel reverse creates a negative-stride view; copy() avoids torch.from_numpy error.
    img = img[:, :, ::-1].copy()                     # BGR→RGB
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


@torch.no_grad()
def infer_single(net, img_bgr: np.ndarray, device: torch.device):
    """
    对单张图像运行推理。
    返回：
        corrected_bgr : np.ndarray  矫正后图像 (512,512,3) uint8
        flow          : np.ndarray  预测 flow   (2,512,512) float32
    """
    tensor = preprocess(img_bgr).to(device)
    flow = net(tensor)                    # eval 模式下直接返回最终 flow
    if isinstance(flow, list):
        flow = flow[-1]

    flow_np = flow[0].cpu().numpy()      # (2, 512, 512)  [0, 1]

    # 构造 remap 坐标（像素坐标）
    map_x = (flow_np[0] * 511).astype(np.float32)
    map_y = (flow_np[1] * 511).astype(np.float32)

    img_resized = cv2.resize(img_bgr, (512, 512))
    corrected   = cv2.remap(img_resized, map_x, map_y,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return corrected, flow_np


def make_result_grid(img_bgr: np.ndarray,
                     corrected: np.ndarray,
                     flow_np: np.ndarray,
                     flow_gt: np.ndarray = None,
                     epe: float = None,
                     title: str = '') -> np.ndarray:
    """
    拼接结果图：[输入 | 矫正结果 | flow可视化 (| GT矫正)]
    """
    H, W = 512, 512
    input_show    = cv2.resize(img_bgr, (W, H))
    correct_show  = corrected
    flow_color    = flow_to_color(flow_np)

    panels = [input_show, correct_show, flow_color]
    labels = ['Input', 'Corrected (epoch80)', 'Flow (pred)']

    if flow_gt is not None:
        map_x_gt = (flow_gt[0] * 511).astype(np.float32)
        map_y_gt = (flow_gt[1] * 511).astype(np.float32)
        gt_corrected = cv2.remap(input_show, map_x_gt, map_y_gt,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        flow_gt_color = flow_to_color(flow_gt)
        panels += [gt_corrected, flow_gt_color]
        labels += ['GT Corrected', 'Flow (GT)']

    # 在每个面板顶部绘制标签
    font      = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    pad        = 28

    annotated = []
    for panel, lbl in zip(panels, labels):
        p = np.zeros((H + pad, W, 3), dtype=np.uint8)
        p[pad:] = panel
        cv2.putText(p, lbl, (6, 18), font, font_scale,
                    (255, 255, 100), thickness, cv2.LINE_AA)
        annotated.append(p)

    grid = np.hstack(annotated)

    # 在整体图顶部写标题 + EPE
    if title or epe is not None:
        info_bar = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
        info_txt = title
        if epe is not None:
            info_txt += f'   EPE={epe:.2f}px'
        cv2.putText(info_bar, info_txt, (10, 24), font, 0.65,
                    (180, 255, 180), 1, cv2.LINE_AA)
        grid = np.vstack([info_bar, grid])

    return grid


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def get_image_list(img_dir: str) -> list:
    imgs = []
    for f in sorted(os.listdir(img_dir)):
        low = f.lower()
        if (low.endswith('.png') or low.endswith('.jpg') or low.endswith('.jpeg')) and not low.endswith('.npy'):
            imgs.append(os.path.join(img_dir, f))
    return imgs


def run_inference(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Device       : {device}')
    print(f'[INFO] Model        : {args.model}')
    print(f'[INFO] Image dir    : {args.img_dir}')
    print(f'[INFO] Output dir   : {args.out_dir}')
    print(f'[INFO] Num images   : {args.num}')

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载模型
    print('[INFO] Loading model...', end=' ', flush=True)
    t0 = time.time()
    net = load_model(args.model, device)
    print(f'done ({time.time()-t0:.2f}s)')

    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'[INFO] Parameters   : {total_params:.2f}M')

    # 收集图像
    img_list = get_image_list(args.img_dir)
    if len(img_list) == 0:
        print(f'[ERROR] No .png images found in {args.img_dir}')
        return
    if args.num > 0:
        img_list = img_list[:args.num]
    print(f'[INFO] Processing {len(img_list)} images...')

    epe_list    = []
    time_list   = []
    result_rows = []   # 用于汇总大图

    for i, img_path in enumerate(img_list):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f'  [WARN] Cannot read {img_path}, skipped.')
            continue

        # 推理计时
        t1 = time.time()
        corrected, flow_np = infer_single(net, img_bgr, device)
        elapsed = time.time() - t1
        time_list.append(elapsed)

        # 读取 GT flow（如果存在）
        gt_path  = img_path + '.npy'
        flow_gt  = None
        epe      = None
        if os.path.exists(gt_path):
            flow_gt = np.load(gt_path).astype(np.float32)
            if flow_gt.shape != (2, 512, 512):
                from skimage.transform import resize as sk_resize
                flow_gt = sk_resize(flow_gt, (2, 512, 512),
                                    anti_aliasing=True, mode='reflect').astype(np.float32)
            epe = compute_epe(flow_np, flow_gt)
            epe_list.append(epe)

        # 保存单张矫正图
        base_name   = os.path.splitext(os.path.basename(img_path))[0]
        save_corrected = os.path.join(args.out_dir, f'{base_name}_corrected.png')
        cv2.imwrite(save_corrected, corrected)

        # 生成对比网格
        title = f'[{i+1}/{len(img_list)}] {base_name}  |  {elapsed*1000:.1f}ms'
        grid  = make_result_grid(img_bgr, corrected, flow_np,
                                 flow_gt=flow_gt, epe=epe, title=title)
        save_grid = os.path.join(args.out_dir, f'{base_name}_compare.png')
        cv2.imwrite(save_grid, grid)

        # 仅将前 args.summary_n 张加入汇总
        if i < args.summary_n:
            result_rows.append(grid)

        epe_str = f'EPE={epe:.2f}px' if epe is not None else 'no GT'
        print(f'  [{i+1:3d}/{len(img_list)}] {base_name:30s}  {elapsed*1000:6.1f}ms  {epe_str}')

    # ── 汇总大图 ──
    if result_rows:
        summary_path = os.path.join(args.out_dir, 'epoch80_summary.png')
        # 每行宽度可能因有无GT而不同，统一宽度
        max_w = max(r.shape[1] for r in result_rows)
        padded = []
        for r in result_rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        summary = np.vstack(padded)
        cv2.imwrite(summary_path, summary)
        print(f'\n[INFO] Summary grid saved: {summary_path}')

    # ── 统计 ──
    print('\n' + '=' * 60)
    print(f'  Epoch 80 Inference Report')
    print('=' * 60)
    print(f'  Images processed  : {len(time_list)}')
    if time_list:
        print(f'  Avg inference time: {np.mean(time_list)*1000:.1f}ms')
        print(f'  FPS               : {1.0/np.mean(time_list):.1f}')
    if epe_list:
        print(f'  Avg EPE           : {np.mean(epe_list):.3f}px')
        print(f'  Min EPE           : {np.min(epe_list):.3f}px')
        print(f'  Max EPE           : {np.max(epe_list):.3f}px')
    else:
        print('  EPE               : N/A (no GT labels found)')
    print(f'  Output dir        : {args.out_dir}')
    print('=' * 60)


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Epoch-80 Inference Script')
    parser.add_argument('--model',     default='output_crackwarp/epoch_80.pth',
                        help='模型权重路径 (default: output_crackwarp/epoch_80.pth)')
    parser.add_argument('--img_dir',   default='underwater_crack_v3',
                        help='测试图像目录 (default: underwater_crack_v3)')
    parser.add_argument('--out_dir',   default='output_crackwarp/epoch80_results',
                        help='结果输出目录')
    parser.add_argument('--num',       type=int, default=20,
                        help='推理图像数量，-1 表示全部 (default: 20)')
    parser.add_argument('--summary_n', type=int, default=10,
                        help='汇总大图包含的图像数量 (default: 10)')
    parser.add_argument('--gpu',       type=int, default=0,
                        help='GPU id (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)
