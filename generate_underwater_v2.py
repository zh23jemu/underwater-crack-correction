# -*- coding: utf-8 -*-
"""
水下裂缝图像数据生成器 v3

修改内容（参考 DocUnet/DewarpNet/RectiNet 等顶会工作）：
  1. [BUG FIX] 标签改为真正的逆映射（forward map -> inverse map via sparse interp）
  2. [BUG FIX] 标签归一化改为分通道归一化到 [0,1]（x/W, y/H）
  3. [BUG FIX] 图像尺寸改为 512x512
  4. 扰动机制：topology(70%) + refraction(30%) 累加位移场（参考 DocUnetC.cpp）
参考文献：
  - DocUNet (Ma et al., CVPR 2018)
  - DewarpNet (Das et al., ICCV 2019)
  - CREASE (Li et al., CVPR 2023)
"""

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import gaussian_filter
import json


# ============================================================
# 工具函数
# ============================================================

def detect_crack_mask(img, thresh=80):
    """Detect crack-like dark thin structures as a binary mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def compute_inverse_map(total_dx, total_dy, h, w, iters=12):
    """
    从正向位移场计算逆映射（快速版本）

    使用 cv2.remap 的迭代近似方法（参考 DewarpNet）：
      正向映射 fwd: src(x,y) -> dst(x+dx, y+dy)
      逆映射 inv:   dst(u,v) -> src位置

    快速实现：直接对位移场取反作为近似逆映射。
    对于小位移（<图像尺寸10%）误差可忽略，适合水下扰动场景。
    """
    base_x, base_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    # 近似逆映射：对正向位移场用双线性插值反向采样。
    # 即在 dst 坐标处采样 src 的负位移。
    inv_map_x = base_x.copy()
    inv_map_y = base_y.copy()

    # 用正向映射的目标位置采样负位移（一次迭代近似）。
    for _ in range(iters):
        samp_dx = cv2.remap(total_dx, inv_map_x, inv_map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        samp_dy = cv2.remap(total_dy, inv_map_x, inv_map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        inv_map_x = np.clip(base_x - samp_dx, 0, w - 1)
        inv_map_y = np.clip(base_y - samp_dy, 0, h - 1)

    return inv_map_x.astype(np.float32), inv_map_y.astype(np.float32)


def smooth_random_field(h, w, sigma_range=(18, 42), amp=6.0):
    """Generate a smooth low-frequency displacement field."""
    fx = np.random.randn(h, w).astype(np.float32)
    fy = np.random.randn(h, w).astype(np.float32)
    sigma = np.random.uniform(*sigma_range)
    fx = gaussian_filter(fx, sigma=sigma, mode='reflect')
    fy = gaussian_filter(fy, sigma=sigma, mode='reflect')
    fx = fx / (np.std(fx) + 1e-6) * amp
    fy = fy / (np.std(fy) + 1e-6) * amp
    return fx.astype(np.float32), fy.astype(np.float32)


def make_underwater_profile(level='mixed'):
    """
    Difficulty profile:
      easy   -> mild geometry + mild optics
      medium -> balanced
      hard   -> stronger geometry + stronger optics
      realistic -> conservative physically plausible distortions
      mixed  -> sampled curriculum mix
    """
    presets = {
        'easy': {
            'n_range': (3, 6),
            'topology_ratio': 0.65,
            'break_strength_range': (5, 14),
            'refraction_amp_range': (2, 8),
            'current_amp': (0.8, 2.8),
            'max_disp_px': 46.0,
            'optical_strength': 0.55,
        },
        'medium': {
            'n_range': (4, 8),
            'topology_ratio': 0.70,
            'break_strength_range': (8, 18),
            'refraction_amp_range': (3, 10),
            'current_amp': (1.2, 3.4),
            'max_disp_px': 62.0,
            'optical_strength': 0.75,
        },
        'hard': {
            'n_range': (6, 10),
            'topology_ratio': 0.72,
            'break_strength_range': (10, 24),
            'refraction_amp_range': (4, 12),
            'current_amp': (1.8, 4.2),
            'max_disp_px': 78.0,
            'optical_strength': 0.95,
        },
        'realistic': {
            'n_range': (4, 7),
            'topology_ratio': 0.68,
            'break_strength_range': (6, 14),
            'refraction_amp_range': (2, 8),
            'current_amp': (0.8, 2.6),
            'max_disp_px': 52.0,
            'optical_strength': 0.62,
        }
    }

    if level == 'mixed':
        sampled = np.random.choice(['easy', 'medium', 'hard'], p=[0.30, 0.50, 0.20]).item()
        out = dict(presets[sampled])
        out['name'] = sampled
        return out

    if level not in presets:
        raise ValueError(f'Unknown profile level: {level}')
    out = dict(presets[level])
    out['name'] = level
    return out


# ============================================================
# 位移场模块：Topology 拓扑扰动
# 对标 DocUnet folds: w = alpha/(dis+alpha)
# ============================================================

def topology_displacement_field(h, w, img, n_breaks=3,
                                 break_strength=35,
                                 alpha_range=(0.05, 0.3)):
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)

    crack_mask = detect_crack_mask(img)
    crack_pixels = np.argwhere(crack_mask > 0)
    if len(crack_pixels) < 10:
        crack_pixels = np.column_stack([
            np.random.randint(h // 4, 3 * h // 4, 50),
            np.random.randint(w // 4, 3 * w // 4, 50)
        ])

    n_breaks = min(n_breaks, len(crack_pixels))
    idxs = np.random.choice(len(crack_pixels), n_breaks, replace=False)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    max_dis = float(np.sqrt(h ** 2 + w ** 2))

    for bp in crack_pixels[idxs]:
        br, bc = float(bp[0]), float(bp[1])
        alpha = np.random.uniform(*alpha_range)
        dis_norm = np.sqrt((xs - bc) ** 2 + (ys - br) ** 2) / max_dis
        w_field = alpha / (dis_norm + alpha)   # DocUnet fold 公式
        angle = np.random.uniform(0, 2 * np.pi)
        side = np.sign((xs - bc) * np.sin(angle) - (ys - br) * np.cos(angle))
        dx += w_field * side * break_strength * np.cos(angle)
        dy += w_field * side * break_strength * np.sin(angle)

    return dx, dy


# ============================================================
# 位移场模块：Refraction 折射扰动
# 对标 DocUnet curves: w = 1 - dis^alpha
# ============================================================

def refraction_displacement_field(h, w, n_waves=3,
                                  amplitude_range=(8, 22),
                                  alpha_range=(0.3, 0.7)):
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    max_dis = float(np.sqrt(h ** 2 + w ** 2))

    for _ in range(n_waves):
        pc = np.array([np.random.randint(0, w),
                       np.random.randint(0, h)], dtype=np.float32)
        alpha = np.random.uniform(*alpha_range)
        amp   = np.random.uniform(*amplitude_range)
        angle = np.random.uniform(0, 2 * np.pi)
        # Keep refraction low-frequency for realistic underwater waviness.
        freq  = np.random.uniform(0.35, 1.25)

        dis_norm = np.sqrt((xs - pc[0]) ** 2 + (ys - pc[1]) ** 2) / max_dis
        w_field  = 1.0 - np.power(np.clip(dis_norm, 0, 1), alpha)  # DocUnet curve 公式
        proj     = xs * np.cos(angle) + ys * np.sin(angle)
        wave     = np.sin(2 * np.pi * freq * proj / max(h, w)
                         + np.random.uniform(0, 2 * np.pi))
        dx += w_field * amp * wave * np.cos(angle)
        dy += w_field * amp * wave * np.sin(angle)

    return dx, dy


# ============================================================
# 耦合扰动核心函数
# ============================================================

def underwater_coupled_v2(img, n=8, topology_ratio=0.7,
                          break_strength_range=(15, 40),
                          refraction_amp_range=(8, 20),
                          current_amp_range=(2.0, 6.0),
                          max_disp_px=160.0,
                          realistic=True):
    """
    水下耦合扰动（对标 DocUnetC.cpp folds_curves_img_vector）。
    累加 n 次位移场：
      - round(n*0.7) 次 topology（对标 folds）
      - round(n*0.3) 次 refraction（对标 curves）
    返回：
      distorted   - 扰动后的图像
      inv_map_x   - 逆映射 x 坐标（用于 cv2.remap）
      inv_map_y   - 逆映射 y 坐标
    """
    h, w = img.shape[:2]
    total_dx = np.zeros((h, w), dtype=np.float32)
    total_dy = np.zeros((h, w), dtype=np.float32)

    n_topology   = round(n * topology_ratio)
    n_refraction = n - n_topology

    for _ in range(n_topology):
        dx, dy = topology_displacement_field(
            h, w, img,
            n_breaks=np.random.randint(1, 4),
            break_strength=np.random.uniform(*break_strength_range),
            alpha_range=(np.random.uniform(0.05, 0.25),
                         np.random.uniform(0.25, 0.45))
        )
        total_dx += dx
        total_dy += dy

    for _ in range(n_refraction):
        dx, dy = refraction_displacement_field(
            h, w,
            n_waves=np.random.randint(2, 5),
            amplitude_range=(np.random.uniform(*refraction_amp_range) * 0.7,
                             np.random.uniform(*refraction_amp_range) * 1.3),
            alpha_range=(np.random.uniform(0.3, 0.5),
                         np.random.uniform(0.5, 0.75))
        )
        total_dx += dx
        total_dy += dy

    # Add low-frequency current drift field for realistic underwater flow.
    cur_amp = np.random.uniform(*current_amp_range)
    cdx, cdy = smooth_random_field(h, w, sigma_range=(28, 62), amp=cur_amp)
    total_dx += cdx
    total_dy += cdy

    # Final smoothing to avoid piecewise-sharp synthetic artifacts.
    sigma_final = np.random.uniform(1.8, 3.6) if realistic else np.random.uniform(1.2, 2.8)
    total_dx = gaussian_filter(total_dx, sigma=sigma_final, mode='reflect').astype(np.float32)
    total_dy = gaussian_filter(total_dy, sigma=sigma_final, mode='reflect').astype(np.float32)

    if realistic:
        # 1) Robustly suppress outliers while preserving smooth global flow.
        mag = np.sqrt(total_dx ** 2 + total_dy ** 2)
        p95 = np.percentile(mag, 95.0)
        if p95 > 1e-6:
            soft_cap = min(max_disp_px, p95 * 1.35)
            scale = np.minimum(1.0, soft_cap / (mag + 1e-6))
            total_dx *= scale
            total_dy *= scale

        # 2) Limit local displacement gradients to avoid "torn" synthetic look.
        grad_target = np.random.uniform(0.55, 0.85)
        for _ in range(2):
            gx_dx = np.gradient(total_dx, axis=1)
            gy_dx = np.gradient(total_dx, axis=0)
            gx_dy = np.gradient(total_dy, axis=1)
            gy_dy = np.gradient(total_dy, axis=0)
            grad_mag = np.sqrt(gx_dx ** 2 + gy_dx ** 2 + gx_dy ** 2 + gy_dy ** 2)
            g95 = np.percentile(grad_mag, 95.0)
            if g95 > grad_target:
                sigma_extra = np.clip((g95 / max(grad_target, 1e-6) - 1.0) * 1.15, 0.2, 1.4)
                total_dx = gaussian_filter(total_dx, sigma=sigma_extra, mode='reflect').astype(np.float32)
                total_dy = gaussian_filter(total_dy, sigma=sigma_extra, mode='reflect').astype(np.float32)

    # Clamp extreme displacement magnitude.
    mag = np.sqrt(total_dx ** 2 + total_dy ** 2) + 1e-6
    scale = np.minimum(1.0, max_disp_px / mag)
    total_dx *= scale
    total_dy *= scale

    # Forward warp: generate distorted image.
    base_x, base_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32)
    )
    distorted = cv2.remap(img,
                          base_x + total_dx,
                          base_y + total_dy,
                          cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

    # *** 关键修复：计算逆映射作为训练标签 ***
    # 逆映射含义：对扭曲图像中每个像素(u,v)，其在原图中的位置(inv_x, inv_y)。
    # 推理时：corrected = cv2.remap(distorted, inv_map_x, inv_map_y)
    inv_map_x, inv_map_y = compute_inverse_map(total_dx, total_dy, h, w)

    return distorted, inv_map_x, inv_map_y


def underwater_optical_noise(img):
    """Legacy underwater optical noise simulation."""
    result = img.astype(np.float32)
    h, w = img.shape[:2]
    brightness = np.random.uniform(-25, 15)
    tint = np.random.uniform(0, 20)
    result[:, :, 0] = np.clip(result[:, :, 0] + brightness - tint * 0.4, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] + brightness + tint * 0.8, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] + brightness - tint * 0.2, 0, 255)
    if np.random.random() < 0.4:
        sigma = np.random.uniform(1.0, 2.5)
        mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            r = np.random.randint(40, min(h, w) // 3)
            ys2, xs2 = np.ogrid[:h, :w]
            mask += np.clip(1.0 - np.sqrt((xs2 - cx) ** 2 + (ys2 - cy) ** 2) / r, 0, 1)
        mask = np.clip(mask, 0, 1)
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma * 3)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - mask) + blurred[:, :, c] * mask
    for _ in range(np.random.randint(0, 5)):
        bx = np.random.randint(0, w)
        by = np.random.randint(0, h)
        br = np.random.randint(2, 7)
        spot = np.random.uniform(30, 90)
        ys2, xs2 = np.ogrid[:h, :w]
        sp = np.clip(1.0 - np.sqrt((xs2 - bx) ** 2 + (ys2 - by) ** 2) / br, 0, 1)
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] + spot * sp, 0, 255)
    return result.astype(np.uint8)


def underwater_optical_noise_v2(img, optical_strength=0.8):
    """
    Physically-inspired underwater degradation:
      attenuation + veiling light + backscatter + depth blur + chromatic shift.
    """
    result = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = yy / max(h - 1, 1)
    tilt = np.random.uniform(-0.15, 0.15) * (xx / max(w - 1, 1) - 0.5)
    depth = np.clip(depth + tilt, 0.0, 1.0)

    beta = np.random.uniform(0.8, 1.8) * optical_strength
    t = np.exp(-beta * depth)[..., None].astype(np.float32)
    ambient_bgr = np.array([0.30, 0.52, 0.40], dtype=np.float32)
    ambient_bgr += np.random.uniform(-0.04, 0.04, size=3).astype(np.float32)
    ambient_bgr = np.clip(ambient_bgr, 0.05, 0.85)
    result = result * t + ambient_bgr * (1.0 - t)

    haze = np.zeros((h, w), dtype=np.float32)
    n_haze = np.random.randint(2, 6)
    for _ in range(n_haze):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        r = np.random.randint(max(20, min(h, w) // 18), max(30, min(h, w) // 4))
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        haze += np.clip(1.0 - rr / max(r, 1), 0.0, 1.0)
    haze = gaussian_filter(np.clip(haze, 0.0, 1.0), sigma=np.random.uniform(4.0, 9.0)).astype(np.float32)
    haze = haze[..., None] * np.random.uniform(0.05, 0.18) * optical_strength
    result = np.clip(result + haze, 0.0, 1.0)

    sigma_blur = np.random.uniform(0.4, 1.8) * optical_strength
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma_blur, sigmaY=sigma_blur)
    blur_alpha = np.clip(depth[..., None] * np.random.uniform(0.15, 0.45), 0.0, 0.6)
    result = result * (1.0 - blur_alpha) + blurred * blur_alpha

    shift = np.random.uniform(-1.5, 1.5) * optical_strength
    M_b = np.float32([[1, 0, shift], [0, 1, 0]])
    M_r = np.float32([[1, 0, -shift], [0, 1, 0]])
    b = cv2.warpAffine(result[:, :, 0], M_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    g = result[:, :, 1]
    r = cv2.warpAffine(result[:, :, 2], M_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    result = np.stack([b, g, r], axis=2)

    noise_sigma = np.random.uniform(0.004, 0.015) * (0.6 + optical_strength)
    result += np.random.randn(h, w, 3).astype(np.float32) * noise_sigma
    gamma = np.random.uniform(0.85, 1.18)
    result = np.clip(result, 0.0, 1.0) ** gamma

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# 批量生成
# ============================================================

def generate_dataset(input_dir, output_dir,
                     target_size=(512, 512),
                     samples_per_image=10,
                     n_range=(4, 10),
                     topology_ratio=0.7,
                     profile_mode='mixed',
                     save_manifest=True):
    """
    批量生成水下裂缝训练数据集。
    标签格式（修复后）：
      label.npy shape = (2, H, W)
        label[0] = inv_map_x / W   归一化到[0,1]，x方向逆映射
        label[1] = inv_map_y / H   归一化到[0,1]，y方向逆映射
    推理时恢复：
      map_x = label[0] * W
      map_y = label[1] * H
      corrected = cv2.remap(distorted, map_x, map_y, cv2.INTER_LINEAR)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    image_files = sorted(
        list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    )
    H, W = target_size
    print(f'Found {len(image_files)} crack images')
    print(f'Target size: {H}x{W}')
    print(f'Expected samples: {len(image_files) * samples_per_image}')
    print(f'Mode: topology {int(topology_ratio * 100)}% + refraction {int((1 - topology_ratio) * 100)}%')
    print(f'Label format: inv_map normalized by (W={W}, H={H})')
    print(f'Profile mode: {profile_mode}')

    total = 0
    profile_hist = {'easy': 0, 'medium': 0, 'hard': 0, 'realistic': 0}
    for img_file in tqdm(image_files, desc='Generating'):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        # 修复：统一 resize 到 512x512。
        img_resized = cv2.resize(img, (W, H))

        for i in range(samples_per_image):
            np.random.seed(hash(img_file.name + str(i)) % (2 ** 31))
            profile = make_underwater_profile(profile_mode)
            profile_hist[profile['name']] += 1
            n = np.random.randint(*profile['n_range'])
            try:
                distorted, inv_x, inv_y = underwater_coupled_v2(
                    img_resized,
                    n=n,
                    topology_ratio=profile['topology_ratio'],
                    break_strength_range=profile['break_strength_range'],
                    refraction_amp_range=profile['refraction_amp_range'],
                    current_amp_range=profile['current_amp'],
                    max_disp_px=profile['max_disp_px'],
                    realistic=True
                )
                distorted = underwater_optical_noise_v2(
                    distorted,
                    optical_strength=profile['optical_strength']
                )
            except Exception as e:
                print(f'Skip {img_file.name}[{i}]: {e}')
                continue

            out_name = f'{img_file.stem}_{i:02d}.png'
            out_path = output_path / out_name
            cv2.imwrite(str(out_path), distorted)

            # *** 修复：标签归一化 x/W, y/H -> [0,1] ***
            label = np.stack([
                inv_x / (W - 1),  # x normalized
                inv_y / (H - 1),  # y normalized
            ], axis=0).astype(np.float32)
            np.save(str(out_path) + '.npy', label)
            total += 1

    print(f'Done! Generated {total} samples -> {output_path}')


def generate_dataset_v2(input_dir, output_dir,
                        target_size=(512, 512),
                        samples_per_image=10,
                        profile_mode='mixed',
                        save_manifest=True):
    """
    Improved curriculum-based underwater synthesis pipeline.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    image_files = sorted(list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')))
    H, W = target_size
    print(f'[v2] Found {len(image_files)} source images')
    print(f'[v2] Target size: {H}x{W}')
    print(f'[v2] Profile mode: {profile_mode}')

    total = 0
    profile_hist = {'easy': 0, 'medium': 0, 'hard': 0, 'realistic': 0}

    for img_file in tqdm(image_files, desc='Generating-v2'):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        img_resized = cv2.resize(img, (W, H))

        for i in range(samples_per_image):
            np.random.seed(hash(img_file.name + str(i)) % (2 ** 31))
            profile = make_underwater_profile(profile_mode)
            profile_hist[profile['name']] += 1

            n = np.random.randint(*profile['n_range'])
            try:
                distorted, inv_x, inv_y = underwater_coupled_v2(
                    img_resized,
                    n=n,
                    topology_ratio=profile['topology_ratio'],
                    break_strength_range=profile['break_strength_range'],
                    refraction_amp_range=profile['refraction_amp_range'],
                    current_amp_range=profile['current_amp'],
                    max_disp_px=profile['max_disp_px'],
                    realistic=True
                )
                distorted = underwater_optical_noise_v2(
                    distorted,
                    optical_strength=profile['optical_strength']
                )
            except Exception as e:
                print(f'[v2] Skip {img_file.name}[{i}]: {e}')
                continue

            out_name = f'{img_file.stem}_{i:02d}.png'
            out_path = output_path / out_name
            cv2.imwrite(str(out_path), distorted)

            label = np.stack([
                inv_x / (W - 1),
                inv_y / (H - 1)
            ], axis=0).astype(np.float32)
            np.save(str(out_path) + '.npy', label)
            total += 1

    print(f'[v2] Done! Generated {total} samples -> {output_path}')
    print(f'[v2] Profile histogram: {profile_hist}')

    if save_manifest:
        manifest = {
            'pipeline': 'generate_dataset_v2',
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'target_size': [int(H), int(W)],
            'samples_per_image': int(samples_per_image),
            'profile_mode': profile_mode,
            'profile_hist': profile_hist,
            'total_generated': int(total),
        }
        with open(output_path / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    generate_dataset_v2(
        input_dir         = './under-crack-images/images/',
        output_dir        = './underwater_crack_v3/',
        target_size       = (512, 512),
        samples_per_image = 10,   # 1037 x 10 = 10370
        profile_mode      = 'realistic',
        save_manifest     = True
    )
