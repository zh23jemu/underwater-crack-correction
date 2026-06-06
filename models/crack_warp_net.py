# -*- coding: utf-8 -*-
"""
CrackWarpNet v2 - 水下裂缝几何畸变矫正网络

改进参考顶刊：
  - Restormer (CVPR 2022):    通道转置注意力 + Gated FFN
  - DocScanner (AAAI 2023):   迭代细化坐标场
  - RAFT (ECCV 2020):         ConvGRU 迭代更新
  - CBAM (ECCV 2018):         空间 + 通道双重注意力
  - NAFNet (ECCV 2022):       SimpleGate 替代激活，去除 BN
  - DewarpNet (ICCV 2019):    粗细两阶段预测
  - Swin Transformer (ICCV 2021): 局部窗口注意力思想

硬件目标：RTX 5090 (31.4GB)，batch_size=8，512×512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 基础模块
# ============================================================

class LayerNorm2d(nn.Module):
    """适配 (B,C,H,W) 格式的 LayerNorm（NAFNet 风格）"""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """
    SimpleGate 激活（NAFNet, ECCV 2022）
    将通道对半分，前半乘以 sigmoid(后半)
    比 GELU 更简单且效果更好
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * torch.sigmoid(x2)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """带残差的双卷积块（保留用于轻量级场景）"""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# ============================================================
# NAFBlock（NAFNet ECCV 2022）
# 去掉 BN，使用 LayerNorm + SimpleGate + CA
# 在图像恢复任务上超越 Restormer
# ============================================================

class NAFBlock(nn.Module):
    """
    NAFNet 基本块
    参考：'Simple Baselines for Image Restoration' (ECCV 2022)
    关键设计：
      1. LayerNorm 替代 BN（对小 batch 更稳定）
      2. SimpleGate 替代 GELU（更少参数，更好效果）
      3. 轻量通道注意力（CA）
    """
    def __init__(self, ch, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_ch  = ch * dw_expand
        ffn_ch = ch * ffn_expand

        # Depth-wise conv branch
        self.conv1 = nn.Conv2d(ch, dw_ch, 1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.conv3 = nn.Conv2d(dw_ch // 2, ch, 1)   # SimpleGate 后通道减半
        self.sg    = SimpleGate()

        # 轻量 Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4 if ch >= 4 else 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 4 if ch >= 4 else 1, ch, 1),
            nn.Sigmoid()
        )

        # FFN branch
        self.norm1 = LayerNorm2d(ch)
        self.norm2 = LayerNorm2d(ch)
        self.fc1   = nn.Conv2d(ch, ffn_ch, 1)
        self.fc2   = nn.Conv2d(ffn_ch // 2, ch, 1)  # SimpleGate 后通道减半

        self.drop  = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta  = nn.Parameter(torch.ones(1, ch, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.ones(1, ch, 1, 1) * 0.1)

    def forward(self, x):
        # Depth-wise attention branch
        inp = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)          # SimpleGate: (B, dw_ch) -> (B, dw_ch//2)
        x = x * self.ca(inp)    # Channel Attention with residual info
        x = self.conv3(x)
        x = self.drop(x)
        y = inp + x * self.beta

        # FFN branch
        x = self.norm2(y)
        x = self.fc1(x)
        x = self.sg(x)          # SimpleGate
        x = self.fc2(x)
        x = self.drop(x)
        return y + x * self.gamma


# ============================================================
# CBAM (ECCV 2018) 空间 + 通道双重注意力
# 在编解码器跳跃连接处插入，过滤无关特征
# ============================================================

class ChannelAttention(nn.Module):
    """CBAM 通道注意力（参考：CBAM ECCV 2018）"""
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(ch, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False)
        )

    def forward(self, x):
        avg = x.mean(dim=[2, 3])           # (B, C)
        mx  = x.amax(dim=[2, 3])           # (B, C)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    参考：'CBAM: Convolutional Block Attention Module' (ECCV 2018)
    在 skip connection 上使用，抑制无关背景特征
    """
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ============================================================
# Restormer 风格通道转置注意力（改进版）
# 参考：Restormer (CVPR 2022)
# ============================================================

class ChannelTransposeAttention(nn.Module):
    """
    改进版 Restormer 通道转置注意力
    - 复杂度 O(C²) 而非 O(H²W²)
    - 适合 bottleneck (32×32) 全局依赖建模
    - 新增：pre-norm 改为 LayerNorm2d（更稳定）
    """
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm = LayerNorm2d(dim)
        self.qkv    = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3,
                                 padding=1, groups=dim * 3, bias=False)
        self.proj   = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv_dw(self.qkv(x_norm))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).reshape(B, C, H, W)
        return x + self.proj(out)


class TransformerBlock(nn.Module):
    """
    Restormer 风格 Transformer 块
    通道注意力 + Gated FFN（GLU）
    """
    def __init__(self, dim, num_heads=8, ffn_expand=2.66):
        super().__init__()
        self.attn = ChannelTransposeAttention(dim, num_heads)
        hidden = int(dim * ffn_expand)
        self.norm = LayerNorm2d(dim)
        self.ffn  = nn.Sequential(
            nn.Conv2d(dim, hidden * 2, 1),
            nn.GLU(dim=1),                  # Gated Linear Unit
            nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x):
        x = self.attn(x)
        x = x + self.ffn(self.norm(x))
        return x


# ============================================================
# 编码器 / 解码器
# ============================================================

class EncoderBlock(nn.Module):
    """
    编码器下采样块：stride=2 深度可分离卷积 + NAFBlock
    NAFBlock 替换原 ResBlock，提升特征质量
    """
    def __init__(self, in_ch, out_ch, num_naf=1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 2, stride=2),   # pixel-unshuffle 风格下采样
            LayerNorm2d(out_ch),
        )
        self.naf = nn.Sequential(
            *[NAFBlock(out_ch) for _ in range(num_naf)]
        )

    def forward(self, x):
        return self.naf(self.down(x))


class DecoderBlock(nn.Module):
    """
    解码器上采样块：双线性×2 + CBAM skip fusion + NAFBlock
    CBAM 在 skip connection 上滤除无关背景
    """
    def __init__(self, in_ch, skip_ch, out_ch, num_naf=1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            LayerNorm2d(out_ch),
        )
        self.cbam = CBAM(skip_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1),
            *[NAFBlock(out_ch) for _ in range(num_naf)]
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.up_conv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=True)
        skip = self.cbam(skip)          # CBAM 过滤 skip 特征
        return self.fuse(torch.cat([x, skip], dim=1))


# ============================================================
# 迭代细化模块（ConvGRU，参考 RAFT ECCV 2020）
# ============================================================

class ConvGRU(nn.Module):
    """卷积 GRU，维护迭代细化隐状态（参考 RAFT）"""
    def __init__(self, input_ch, hidden_ch):
        super().__init__()
        self.update_gate = nn.Sequential(
            nn.Conv2d(input_ch + hidden_ch, hidden_ch, 3, padding=1),
            nn.Sigmoid()
        )
        self.reset_gate  = nn.Sequential(
            nn.Conv2d(input_ch + hidden_ch, hidden_ch, 3, padding=1),
            nn.Sigmoid()
        )
        self.new_gate    = nn.Sequential(
            nn.Conv2d(input_ch + hidden_ch, hidden_ch, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, h):
        xh  = torch.cat([x, h], dim=1)
        z   = self.update_gate(xh)
        r   = self.reset_gate(xh)
        xrh = torch.cat([x, r * h], dim=1)
        n   = self.new_gate(xrh)
        return (1 - z) * h + z * n


class FlowRefinementModule(nn.Module):
    """
    迭代坐标场细化模块（改进版）
    - 使用 NAFBlock 替代原 ResBlock，更强特征提取
    - ConvGRU 维护跨迭代状态（参考 RAFT ECCV 2020）
    - 输出残差 delta_flow，迭代累加
    """
    def __init__(self, feat_ch=128, hidden_ch=128):
        super().__init__()
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(feat_ch * 2 + 2, hidden_ch, 1),
            NAFBlock(hidden_ch),
            NAFBlock(hidden_ch),
        )
        self.gru = ConvGRU(hidden_ch, hidden_ch)
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_ch, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 2, 1)
        )

    def forward(self, feat_orig, feat_warp, flow, hidden):
        x = torch.cat([feat_orig, feat_warp, flow], dim=1)
        x = self.diff_encoder(x)
        hidden = self.gru(x, hidden)
        delta  = self.flow_head(hidden)
        return delta, hidden


# ============================================================
# CrackWarpNet v2 主网络
# ============================================================

class CrackWarpNet(nn.Module):
    """
    水下裂缝几何畸变矫正网络 v2

    架构改进：
      1. EncoderBlock 使用 NAFBlock（替代 ResBlock）
      2. Bottleneck 使用更多 TransformerBlock（4块）
      3. DecoderBlock 在 skip connection 上插入 CBAM
      4. 迭代细化次数从 3 增至 6（RTX5090 显存充足）
      5. 粗预测头加入 NAFBlock 增强表达

    训练输出：[coarse_flow, iter1, ..., iter_n]
    推理输出：iter_n (最终细化结果)
    所有 flow 值域 [0,1]（归一化逆映射坐标）
    """
    def __init__(self,
                 in_ch=3,
                 base_ch=64,
                 num_heads=8,
                 n_iter=6,
                 n_transformer_blocks=4,
                 drop_out_rate=0.1,
                 refine_delta_scale=0.10,
                 bounded_refine_update=False):
        super().__init__()
        self.n_iter = n_iter
        self.refine_delta_scale = float(refine_delta_scale)
        self.bounded_refine_update = bool(bounded_refine_update)

        # 通道数：64→128→256→512→512
        ch = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*8]

        # ── 编码器（每级 2 个 NAFBlock）──
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch[0], 3, padding=1, bias=False),
            LayerNorm2d(ch[0]),
            NAFBlock(ch[0]),
        )
        self.enc1 = EncoderBlock(ch[0], ch[1], num_naf=2)   # 512→256
        self.enc2 = EncoderBlock(ch[1], ch[2], num_naf=2)   # 256→128
        self.enc3 = EncoderBlock(ch[2], ch[3], num_naf=2)   # 128→64
        self.enc4 = EncoderBlock(ch[3], ch[4], num_naf=2)   # 64→32

        # ── Bottleneck Transformer（4块，32×32全局依赖）──
        self.transformer = nn.Sequential(
            *[TransformerBlock(ch[4], num_heads)
              for _ in range(n_transformer_blocks)]
        )

        # ── 解码器（每级 CBAM skip + NAFBlock）──
        self.dec4 = DecoderBlock(ch[4], ch[3], ch[3], num_naf=2)  # 32→64
        self.dec3 = DecoderBlock(ch[3], ch[2], ch[2], num_naf=2)  # 64→128
        self.dec2 = DecoderBlock(ch[2], ch[1], ch[1], num_naf=2)  # 128→256
        self.dec1 = DecoderBlock(ch[1], ch[0], ch[0], num_naf=2)  # 256→512

        # ── 粗预测头 ──
        self.coarse_head = nn.Sequential(
            NAFBlock(ch[0]),
            NAFBlock(ch[0]),
            nn.Conv2d(ch[0], 2, 1),
            nn.Sigmoid()
        )

        # ── 迭代细化特征提取（轻量，128×128 低分辨率）──
        refine_ch = ch[1]   # 128
        self.refine_feat_enc = nn.Sequential(
            nn.Conv2d(in_ch, ch[0], 3, padding=1, bias=False),
            LayerNorm2d(ch[0]),
            NAFBlock(ch[0]),
            nn.Conv2d(ch[0], refine_ch, 1),
            NAFBlock(refine_ch),
        )
        self.refine     = FlowRefinementModule(feat_ch=refine_ch, hidden_ch=refine_ch)
        self._hidden_ch = refine_ch

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _warp(self, img, flow):
        """
        用归一化坐标场 [0,1] 对图像做双线性采样
        flow: (B, 2, H, W)
        """
        grid_x = flow[:, 0] * 2 - 1
        grid_y = flow[:, 1] * 2 - 1
        grid   = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)
        return F.grid_sample(img, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # ── 编码 ──
        s0 = self.stem(x)          # (B, 64,  512, 512)
        s1 = self.enc1(s0)         # (B, 128, 256, 256)
        s2 = self.enc2(s1)         # (B, 256, 128, 128)
        s3 = self.enc3(s2)         # (B, 512, 64,  64)
        s4 = self.enc4(s3)         # (B, 512, 32,  32)

        # ── Bottleneck Transformer ──
        bt = self.transformer(s4)  # (B, 512, 32, 32)

        # ── 解码 ──
        d4 = self.dec4(bt, s3)     # (B, 512, 64,  64)
        d3 = self.dec3(d4, s2)     # (B, 256, 128, 128)
        d2 = self.dec2(d3, s1)     # (B, 128, 256, 256)
        d1 = self.dec1(d2, s0)     # (B, 64,  512, 512)

        # ── 粗预测 ──
        coarse_flow = self.coarse_head(d1)   # (B, 2, 512, 512) [0,1]

        # ── 迭代细化（128×128 低分辨率）──
        refine_size = (128, 128)
        x_lr      = F.interpolate(x, size=refine_size,
                                  mode='bilinear', align_corners=True)
        feat_orig = self.refine_feat_enc(x_lr)   # (B, 128, 128, 128)

        flow_lr = F.interpolate(coarse_flow, size=refine_size,
                                mode='bilinear', align_corners=True)
        h = torch.zeros(B, self._hidden_ch, *refine_size,
                        device=x.device, dtype=x.dtype)

        flows_up  = [coarse_flow]
        flow_iter = flow_lr

        for _ in range(self.n_iter):
            warped_lr = self._warp(x_lr, flow_iter)
            feat_warp = self.refine_feat_enc(warped_lr)
            delta, h  = self.refine(feat_orig, feat_warp, flow_iter, h)
            if self.bounded_refine_update:
                # New bounded update for future re-training.
                delta = torch.tanh(delta) * self.refine_delta_scale
                flow_iter = torch.clamp(flow_iter + delta, 0.0, 1.0)
            else:
                # Legacy behavior: keep compatibility for existing checkpoints.
                flow_iter = torch.clamp(flow_iter + delta, 0.0, 1.0)
            flow_full = F.interpolate(flow_iter, size=(H, W),
                                      mode='bilinear', align_corners=True)
            flows_up.append(flow_full)

        if self.training:
            return flows_up   # [coarse, iter1, ..., iter_n]
        else:
            return flows_up[-1]


# ============================================================
# EMA 工具（Mean Teacher 风格）
# 参考：'Mean teachers are better role models' (NeurIPS 2017)
# ============================================================

class ModelEMA:
    """
    指数移动平均（EMA）：维护一份参数平均版本用于推理
    提升泛化，尤其在小数据集上效果显著
    """
    def __init__(self, model, decay=0.9999):
        import copy
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd  = model.state_dict()
        emsd = self.module.state_dict()
        for k in emsd:
            emsd[k].mul_(self.decay).add_(msd[k], alpha=1.0 - self.decay)
        self.module.load_state_dict(emsd)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


# ============================================================
# 模型工厂
# ============================================================

def build_crack_warp_net(pretrained=False):
    """
    RTX 5090 (31.4GB) configuration.
    """
    model = CrackWarpNet(
        in_ch=3,
        base_ch=56,
        num_heads=8,
        n_iter=3,
        n_transformer_blocks=2,
        drop_out_rate=0.1,
        refine_delta_scale=0.10,
        bounded_refine_update=False,
    )
    return model


if __name__ == '__main__':
    import time
    model = build_crack_warp_net()
    model.eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Parameters: {total:.2f}M')
    x = torch.randn(1, 3, 512, 512)
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    print(f'Inference: {time.time()-t0:.3f}s')
    print(f'Output shape: {out.shape}  Range: [{out.min():.3f}, {out.max():.3f}]')
