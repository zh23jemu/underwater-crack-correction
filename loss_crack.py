# -*- coding: utf-8 -*-
"""
Crack-centric loss design for CrackWarpNet.

Core idea:
- Keep global supervision for stable convergence.
- Use a crack attention mask to up-weight thin crack structures.
- Add crack-structure terms (gradient/frequency) to avoid over-smoothing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', patch_factor=1, ave_spectrum=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum

    def tensor2freq(self, x):
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, 'Image size must be divisible by patch_factor'

        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])
        y = torch.stack(patch_list, 1)
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def forward(self, pred, target):
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)
        loss = F.l1_loss(pred_freq, target_freq, reduction=self.reduction)
        return loss * self.loss_weight


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._create_window(window_size))

    @staticmethod
    def _create_window(size):
        sigma = 1.5
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x, y):
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        c = x.size(1)
        window = self.window.expand(c, 1, -1, -1)
        mu_x = F.conv2d(x, window, padding=self.window_size // 2, groups=c)
        mu_y = F.conv2d(y, window, padding=self.window_size // 2, groups=c)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y
        sig_x = F.conv2d(x * x, window, padding=self.window_size // 2, groups=c) - mu_x2
        sig_y = F.conv2d(y * y, window, padding=self.window_size // 2, groups=c) - mu_y2
        sig_xy = F.conv2d(x * y, window, padding=self.window_size // 2, groups=c) - mu_xy
        num = (2 * mu_xy + c1) * (2 * sig_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2)
        return torch.clamp((1.0 - num / den) / 2.0, 0.0, 1.0)

    def forward(self, pred, target):
        return self._ssim(pred, target).mean()


def flow_to_grid(flow):
    grid_x = flow[:, 0] * 2.0 - 1.0
    grid_y = flow[:, 1] * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


class PhotometricWarpLoss(nn.Module):
    def __init__(self, charbonnier_eps=1e-3):
        super().__init__()
        self.char_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.ssim = SSIMLoss(window_size=11)

    def forward(self, pred_flow, gt_flow, img, w_ssim=0.1):
        pred_grid = flow_to_grid(pred_flow)
        gt_grid = flow_to_grid(gt_flow)
        pred_img = F.grid_sample(img, pred_grid, mode='bilinear', padding_mode='border', align_corners=True)
        with torch.no_grad():
            gt_img = F.grid_sample(img, gt_grid, mode='bilinear', padding_mode='border', align_corners=True)
        photo = self.char_loss(pred_img, gt_img)
        if w_ssim > 0:
            photo = photo + w_ssim * self.ssim(pred_img, gt_img)
        return photo


class CrackMaskEstimator(nn.Module):
    """
    Build a soft crack-attention map from the distorted input image.
    No extra annotation is required.
    """

    def __init__(self, topk=0.08, temperature=0.07):
        super().__init__()
        self.topk = float(topk)
        self.temperature = float(temperature)

    @staticmethod
    def _min_pool2d(x, k):
        return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)

    @staticmethod
    def _sobel(gray):
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)

    def forward(self, img):
        # img: (B,3,H,W), RGB [0,1]
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        # dark-line prior
        mu = gray.mean(dim=(2, 3), keepdim=True)
        std = gray.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        dark = ((mu - gray) / (1.5 * std)).clamp_min(0.0)

        # black-hat response for thin dark structures
        close = self._min_pool2d(F.max_pool2d(gray, kernel_size=7, stride=1, padding=3), 7)
        blackhat = (close - gray).clamp_min(0.0)

        # edge response
        edge = self._sobel(gray)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))

        score = 0.50 * dark + 0.35 * blackhat + 0.15 * edge

        b, _, h, w = score.shape
        flat = score.view(b, -1)
        q = 1.0 - self.topk
        q = min(max(q, 0.5), 0.995)
        thresh = torch.quantile(flat, q=q, dim=1, keepdim=True).view(b, 1, 1, 1)

        soft = torch.sigmoid((score - thresh) / self.temperature)
        soft = F.max_pool2d(soft, kernel_size=3, stride=1, padding=1)
        return soft.clamp(0.0, 1.0)


class FlowGradientConsistencyLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _grad(flow):
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        return dx, dy

    def forward(self, pred, target, roi_weight):
        pdx, pdy = self._grad(pred)
        tdx, tdy = self._grad(target)

        wx = torch.maximum(roi_weight[:, :, :, 1:], roi_weight[:, :, :, :-1])
        wy = torch.maximum(roi_weight[:, :, 1:, :], roi_weight[:, :, :-1, :])

        lx = torch.sqrt((pdx - tdx) ** 2 + self.eps ** 2)
        ly = torch.sqrt((pdy - tdy) ** 2 + self.eps ** 2)

        loss_x = (lx * wx).sum() / wx.sum().clamp_min(1.0)
        loss_y = (ly * wy).sum() / wy.sum().clamp_min(1.0)
        return 0.5 * (loss_x + loss_y)


class EdgeAwareSmoothnessLoss(nn.Module):
    def forward(self, flow, img, protect_mask=None, protect_factor=0.25):
        flow_dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
        flow_dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()

        img_gray = img.mean(dim=1, keepdim=True)
        img_dx = (img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1]).abs()
        img_dy = (img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :]).abs()

        w_x = torch.exp(-img_dx * 10.0)
        w_y = torch.exp(-img_dy * 10.0)

        if protect_mask is not None:
            mx = torch.maximum(protect_mask[:, :, :, 1:], protect_mask[:, :, :, :-1])
            my = torch.maximum(protect_mask[:, :, 1:, :], protect_mask[:, :, :-1, :])
            # Lower smoothness penalty at crack regions to avoid over-smoothing thin branches.
            w_x = w_x * (1.0 - (1.0 - protect_factor) * mx)
            w_y = w_y * (1.0 - (1.0 - protect_factor) * my)

        loss_x = (flow_dx * w_x).mean()
        loss_y = (flow_dy * w_y).mean()
        return 0.5 * (loss_x + loss_y)


class FoldingPenalty(nn.Module):
    def forward(self, flow):
        u = flow[:, 0]
        v = flow[:, 1]

        du_dx = (u[:, :, 2:] - u[:, :, :-2]) / 2.0
        du_dy = (u[:, 2:, :] - u[:, :-2, :]) / 2.0
        dv_dx = (v[:, :, 2:] - v[:, :, :-2]) / 2.0
        dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / 2.0

        du_dx = du_dx[:, 1:-1, :]
        dv_dy = dv_dy[:, :, 1:-1]
        du_dy = du_dy[:, :, 1:-1]
        dv_dx = dv_dx[:, 1:-1, :]

        det = du_dx * dv_dy - du_dy * dv_dx
        return F.relu(-det).mean()


class CrackWarpLoss(nn.Module):
    def __init__(
        self,
        w_coord=1.0,
        w_smooth=0.5,
        w_fold=0.05,
        w_photo=0.0,
        w_ssim=0.1,
        w_freq=0.1,
        gamma=0.8,
        charbonnier_eps=1e-3,
        w_crack_coord=1.2,
        w_crack_grad=0.25,
        w_crack_freq=0.15,
        crack_boost=8.0,
        crack_topk=0.08,
        crack_smooth_factor=0.25,
    ):
        super().__init__()
        self.w_coord = w_coord
        self.w_smooth = w_smooth
        self.w_fold = w_fold
        self.w_photo = w_photo
        self.w_ssim = w_ssim
        self.w_freq = w_freq
        self.gamma = gamma

        self.w_crack_coord = w_crack_coord
        self.w_crack_grad = w_crack_grad
        self.w_crack_freq = w_crack_freq
        self.crack_boost = crack_boost
        self.crack_smooth_factor = crack_smooth_factor

        self.char_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.smooth_loss = EdgeAwareSmoothnessLoss()
        self.fold_loss = FoldingPenalty()
        self.freq_loss = FrequencyLoss(loss_weight=1.0)
        self.photo_loss = PhotometricWarpLoss(charbonnier_eps=charbonnier_eps)

        self.crack_masker = CrackMaskEstimator(topk=crack_topk)
        self.grad_consistency = FlowGradientConsistencyLoss(eps=charbonnier_eps)

    def _weighted_charbonnier(self, pred, target, weight):
        diff = torch.sqrt((pred - target) ** 2 + self.char_loss.eps ** 2)
        w = weight.expand_as(diff)
        return (diff * w).sum() / w.sum().clamp_min(1.0)

    def _masked_charbonnier(self, pred, target, mask):
        diff = torch.sqrt((pred - target) ** 2 + self.char_loss.eps ** 2)
        m = mask.expand_as(diff)
        return (diff * m).sum() / m.sum().clamp_min(1.0)

    def forward(self, flows, label, img=None):
        if not isinstance(flows, list):
            flows = [flows]

        n = len(flows)
        final_flow = flows[-1]
        lbl_full = F.interpolate(label, size=final_flow.shape[2:], mode='bilinear', align_corners=True)

        crack_mask = None
        crack_weight = None
        if img is not None:
            img_full = F.interpolate(img, size=final_flow.shape[2:], mode='bilinear', align_corners=True)
            crack_mask = self.crack_masker(img_full).detach()
            crack_weight = (1.0 + self.crack_boost * crack_mask).detach()
        else:
            img_full = None

        coord_loss = 0.0
        for i, flow in enumerate(flows):
            w = self.gamma ** (n - 1 - i)
            lbl_i = F.interpolate(label, size=flow.shape[2:], mode='bilinear', align_corners=True)
            if crack_weight is not None:
                cw = F.interpolate(crack_weight, size=flow.shape[2:], mode='bilinear', align_corners=True)
                coord_i = self._weighted_charbonnier(flow, lbl_i, cw)
            else:
                coord_i = self.char_loss(flow, lbl_i)
            coord_loss = coord_loss + w * coord_i

        if img_full is not None:
            smooth = self.smooth_loss(
                final_flow,
                img_full,
                protect_mask=crack_mask,
                protect_factor=self.crack_smooth_factor,
            )
        else:
            dx = (final_flow[:, :, :, 1:] - final_flow[:, :, :, :-1]).abs().mean()
            dy = (final_flow[:, :, 1:, :] - final_flow[:, :, :-1, :]).abs().mean()
            smooth = 0.5 * (dx + dy)

        fold = self.fold_loss(final_flow)
        freq = self.freq_loss(final_flow, lbl_full)

        if img_full is not None and self.w_photo > 0:
            photo = self.photo_loss(final_flow, lbl_full, img_full, w_ssim=self.w_ssim)
        else:
            photo = final_flow.new_zeros(())

        if crack_mask is not None:
            crack_coord = self._masked_charbonnier(final_flow, lbl_full, crack_mask)
            crack_grad = self.grad_consistency(final_flow, lbl_full, crack_mask)
            crack_freq = self.freq_loss(final_flow * crack_mask, lbl_full * crack_mask)
            crack_ratio = crack_mask.mean()
        else:
            zero = final_flow.new_zeros(())
            crack_coord = zero
            crack_grad = zero
            crack_freq = zero
            crack_ratio = zero

        total = (
            self.w_coord * coord_loss
            + self.w_smooth * smooth
            + self.w_fold * fold
            + self.w_photo * photo
            + self.w_freq * freq
            + self.w_crack_coord * crack_coord
            + self.w_crack_grad * crack_grad
            + self.w_crack_freq * crack_freq
        )

        loss_dict = {
            'total': total.item(),
            'coord': coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss,
            'smooth': smooth.item() if isinstance(smooth, torch.Tensor) else smooth,
            'fold': fold.item(),
            'photo': photo.item() if isinstance(photo, torch.Tensor) else photo,
            'freq': freq.item(),
            'crack_coord': crack_coord.item(),
            'crack_grad': crack_grad.item(),
            'crack_freq': crack_freq.item(),
            'crack_ratio': crack_ratio.item(),
        }
        return total, loss_dict


if __name__ == '__main__':
    criterion = CrackWarpLoss(
        w_coord=1.0,
        w_smooth=0.1,
        w_fold=0.01,
        w_photo=0.1,
        w_ssim=0.1,
        w_freq=0.1,
        w_crack_coord=1.2,
        w_crack_grad=0.25,
        w_crack_freq=0.15,
    )
    flows = [torch.sigmoid(torch.randn(2, 2, 512, 512)) for _ in range(4)]
    label = torch.sigmoid(torch.randn(2, 2, 512, 512))
    img = torch.rand(2, 3, 512, 512)

    loss, info = criterion(flows, label, img=img)
    print('Loss:', loss.item())
    print('Components:', {k: f'{v:.5f}' for k, v in info.items()})
