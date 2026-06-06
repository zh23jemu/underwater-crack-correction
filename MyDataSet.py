# -*- coding: utf-8 -*-
"""
数据集与增强模块 v2

数据增强参考：
  - DocScanner (AAAI 2023): 几何增强需同步变换 flow label
  - DewarpNet (ICCV 2019): 颜色/亮度增强
  - RAFT (ECCV 2020): 随机裁剪 + 遮挡
  - SimCLR (ICML 2020): 颜色抖动策略
  - 水下图像专项: 颜色偏移、散射模拟、模糊
"""
import cv2
import numpy as np
import torch
import torch.utils.data as Data
import os
import random
from PIL import Image
from natsort import natsorted
from skimage.transform import resize
from torchvision import transforms
import pathlib


# ──────────────────────────────────────────────────────────────
# 文件列表工具
# ──────────────────────────────────────────────────────────────

def get_file_list(folder_path: str, p_postfix=None, sub_dir: bool = False) -> list:
    if p_postfix is None:
        p_postfix = ['.png']
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = []
    if sub_dir:
        for rootdir, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(rootdir, file)
                for p in p_postfix:
                    if os.path.isfile(file_path) and (file_path.endswith(p) or p == '.*'):
                        file_list.append(file_path)
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            for p in p_postfix:
                if os.path.isfile(file_path) and (file_path.endswith(p) or p == '.*'):
                    file_list.append(file_path)
    return natsorted(file_list)


# ──────────────────────────────────────────────────────────────
# 水下图像专项增强（仅作用于 image，不影响 flow label）
# ──────────────────────────────────────────────────────────────

class UnderwaterColorShift:
    """
    水下颜色偏移模拟：红色衰减、蓝绿色增强
    参考: 'Underwater Image Enhancement via Medium Transmission-Guided
           Multi-Color Space Embedding' (TIP 2021)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """img: (H, W, 3) float32 [0,1] BGR"""
        if random.random() > self.p:
            return img
        img = img.copy()
        # 衰减红色通道 (BGR: index 2)
        img[:, :, 2] *= random.uniform(0.4, 0.85)
        # 增强蓝色通道 (BGR: index 0)
        img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(1.0, 1.4), 0, 1)
        return img


class UnderwaterScatter:
    """
    水下散射/雾化模拟
    I = I_clear * t + B * (1-t),  t ~ Uniform(0.4, 0.85)
    参考: 'Underwater Image Restoration via Physics-Based Unpaired Training' (CVPR 2023)
    """
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        t = random.uniform(0.4, 0.85)
        B = np.array([random.uniform(0.6, 0.9),   # B
                      random.uniform(0.5, 0.8),   # G
                      random.uniform(0.05, 0.3)], dtype=np.float32)  # R
        img = img * t + B * (1.0 - t)
        return np.clip(img, 0, 1).astype(np.float32)


class GaussianNoise:
    """高斯噪声（模拟水下传感器噪声）"""
    def __init__(self, p=0.4, sigma_range=(0.01, 0.04)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.randn(*img.shape).astype(np.float32) * sigma
        return np.clip(img + noise, 0, 1)


class RandomGaussianBlur:
    """随机高斯模糊（模拟水下光学模糊）"""
    def __init__(self, p=0.3, kernel_range=(3, 7)):
        self.p = p
        self.kernel_range = kernel_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        k = random.choice(range(self.kernel_range[0], self.kernel_range[1] + 1, 2))
        sigma = random.uniform(0.5, 2.0)
        img_u8 = (img * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_u8, (k, k), sigma)
        return (blurred / 255.0).astype(np.float32)


class ColorJitter:
    """
    颜色抖动（仅作用于 image）
    参考：SimCLR (ICML 2020) 的颜色增强策略
    """
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4,
                 saturation=0.3, hue=0.1):
        self.p = p
        self.jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        # cv2 BGR -> PIL RGB
        img_pil = Image.fromarray((img[:, :, ::-1] * 255).astype(np.uint8))
        img_pil = self.jitter(img_pil)
        # PIL RGB -> cv2 BGR
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        return img_np[:, :, ::-1].copy()


# ──────────────────────────────────────────────────────────────
# 几何增强（image + flow label 同步变换）
# ──────────────────────────────────────────────────────────────

class RandomHorizontalFlip:
    """
    水平翻转：flow[0] (x坐标) -> 1 - flow[0]
    参考：DewarpNet (ICCV 2019) 对称增强
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, label: np.ndarray):
        if random.random() > self.p:
            return img, label
        img = img[:, ::-1, :].copy()
        label = label[:, :, ::-1].copy()
        label[0] = 1.0 - label[0]
        return img, label


class RandomVerticalFlip:
    """
    垂直翻转：flow[1] (y坐标) -> 1 - flow[1]
    """
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img: np.ndarray, label: np.ndarray):
        if random.random() > self.p:
            return img, label
        img = img[::-1, :, :].copy()
        label = label[:, ::-1, :].copy()
        label[1] = 1.0 - label[1]
        return img, label


class RandomRotate90:
    """
    随机90/180/270度旋转，同步变换 flow label
    90度旋转时：(x,y) -> (1-y, x)
    参考：DocScanner (AAAI 2023) 几何增强
    """
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img: np.ndarray, label: np.ndarray):
        if random.random() > self.p:
            return img, label
        k = random.choice([1, 2, 3])
        img = np.rot90(img, k).copy()
        lx, ly = label[0].copy(), label[1].copy()
        for _ in range(k):
            # 单次90度旋转坐标变换: (x,y) -> (1-y, x)
            rot_x = np.rot90(lx, 1)
            rot_y = np.rot90(ly, 1)
            new_x = rot_y
            new_y = 1.0 - rot_x
            lx, ly = new_x.copy(), new_y.copy()
        label = np.stack([lx, ly], axis=0)
        return img, label


class RandomErasing:
    """
    随机矩形遮挡（仅作用于 image，模拟水下遮挡物）
    参考：Random Erasing (AAAI 2020)
    """
    def __init__(self, p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.0)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img
        H, W = img.shape[:2]
        area = H * W
        erase_area = area * random.uniform(*self.scale)
        ar = random.uniform(*self.ratio)
        eh = int(round((erase_area * ar) ** 0.5))
        ew = int(round((erase_area / ar) ** 0.5))
        eh, ew = min(eh, H), min(ew, W)
        y0 = random.randint(0, H - eh)
        x0 = random.randint(0, W - ew)
        img = img.copy()
        # 填充随机颜色（模拟水下随机干扰）
        img[y0:y0+eh, x0:x0+ew, :] = np.random.uniform(0, 0.5, (eh, ew, 3)).astype(np.float32)
        return img


# ──────────────────────────────────────────────────────────────
# 联合增强管线
# ──────────────────────────────────────────────────────────────

class CrackAugmentation:
    """
    训练期专用联合增强管线
    几何增强同步作用于 (image, label)，颜色增强仅作用于 image
    """
    def __init__(self, training=True):
        self.training = training
        # 几何增强
        self.hflip   = RandomHorizontalFlip(p=0.5)
        self.vflip   = RandomVerticalFlip(p=0.3)
        self.rot90   = RandomRotate90(p=0.2)
        # 外观增强（仅 image）
        self.color   = ColorJitter(p=0.8)
        self.uw_color = UnderwaterColorShift(p=0.5)
        self.scatter = UnderwaterScatter(p=0.3)
        self.noise   = GaussianNoise(p=0.4)
        self.blur    = RandomGaussianBlur(p=0.3)
        self.erase   = RandomErasing(p=0.3)

    def __call__(self, img: np.ndarray, label: np.ndarray):
        """
        img:   (H, W, 3) float32 [0,1] BGR
        label: (2, H, W) float32 [0,1]
        """
        if not self.training:
            return img, label

        # 几何增强（image + label 同步）
        img, label = self.hflip(img, label)
        img, label = self.vflip(img, label)
        img, label = self.rot90(img, label)

        # 外观增强（仅 image）
        img = self.color(img)
        img = self.uw_color(img)
        img = self.scatter(img)
        img = self.noise(img)
        img = self.blur(img)
        img = self.erase(img)

        return img, label


# ──────────────────────────────────────────────────────────────
# 数据集
# ──────────────────────────────────────────────────────────────

class ImageData(Data.Dataset):
    """
    水下裂缝矫正数据集

    返回：
      image: (3, 512, 512) float32 tensor [0,1]
      label: (2, 512, 512) float32 tensor [0,1]  归一化逆映射坐标场
    """
    def __init__(self, img_root, training=True):
        self.image_path = get_file_list(img_root, p_postfix=['.png'], sub_dir=True)
        self.image_path = [x for x in self.image_path
                           if pathlib.Path(x).stat().st_size > 0]
        paired_images = []
        paired_labels = []
        for img_path in self.image_path:
            lbl_path = img_path + '.npy'
            if pathlib.Path(lbl_path).exists() and pathlib.Path(lbl_path).stat().st_size > 0:
                paired_images.append(img_path)
                paired_labels.append(lbl_path)
        self.image_path = paired_images
        self.label_path = paired_labels
        self.augment = CrackAugmentation(training=training)
        self.training = training

    def __getitem__(self, index):
        # 读取图像 (BGR float32)
        image = cv2.imread(self.image_path[index])
        if image is None:
            # 跳过损坏文件
            return self.__getitem__((index + 1) % len(self))
        image = cv2.resize(image, (512, 512)).astype(np.float32) / 255.0  # (H,W,3)

        # 读取 flow label
        label = np.load(self.label_path[index])  # (2, H, W)
        if label.shape != (2, 512, 512):
            label = resize(label, (2, 512, 512),
                           anti_aliasing=True, mode='reflect')
        label = np.clip(label.astype(np.float32), 0.0, 1.0)

        # 数据增强
        image, label = self.augment(image, label)

        # numpy -> tensor
        # image: (H,W,3) BGR -> (3,H,W) RGB
        image = torch.from_numpy(
            image[:, :, ::-1].transpose(2, 0, 1).copy()
        ).float()
        label = torch.from_numpy(label).float()
        return image, label

    def __len__(self):
        return len(self.image_path)


# ──────────────────────────────────────────────────────────────
# Mixup（在 DataLoader 之外的 collate 层实现）
# 参考：mixup: Beyond Empirical Risk Minimization (ICLR 2018)
# ──────────────────────────────────────────────────────────────

def mixup_data(images, labels, alpha=0.2):
    """
    批量 Mixup 增强
    images: (B, 3, H, W)
    labels: (B, 2, H, W)
    """
    if alpha <= 0:
        return images, labels
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    idx = torch.randperm(B, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels
