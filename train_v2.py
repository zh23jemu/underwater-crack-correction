#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CrackWarpNet 训练脚本 v3

优化改进（基于80epoch训练数据分析）：
  - 问题1: 模型容量不足 → base_ch=64 (26M参数)
  - 问题2: batch=4梯度噪声大 → batch=1, lr=2e-5
  - 问题3: 数据增强过强 → w_smooth=0.1, Mixup推迟到ep20且概率降至30%
  - 问题5: LR衰减过快 → CosineAnnealingWarmRestarts (T0=25, T_mult=2)
  - 额外: checkpoint每5epoch保存一次，防止服务器重启丢失进度

参考文献：
  - RAFT (ECCV 2020), NAFNet (ECCV 2022), DocScanner (AAAI 2023)
  - SGDR: Stochastic Gradient Descent with Warm Restarts (ICLR 2017)
  - Mean Teacher (NeurIPS 2017)
"""
import torch
import torch.utils.data as Data
import numpy as np
import random
from torch.amp import autocast, GradScaler
from MyDataSet import ImageData, mixup_data
from models.crack_warp_net import build_crack_warp_net, ModelEMA
from loss_crack import CrackWarpLoss
import time, os, shutil
import config_crack as config
from datetime import datetime

print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM: {:.1f}GB'.format(
        torch.cuda.get_device_properties(0).total_memory / 1024**3))


def compute_epe(pred, gt):
    """端点误差 EPE（像素），scale=512"""
    try:
        diff = (pred - gt) * 511.0
        return torch.sqrt((diff ** 2).sum(dim=1)).mean().item()
    except Exception:
        return float('nan')


def load_initial_checkpoint(net, checkpoint_path, device, log_file):
    """
    从已有模型权重初始化网络，用于短程 fine-tune。

    当前训练脚本保存的是模型 state_dict，不包含 optimizer、scheduler 或 scaler。
    因此这里不是严格断点续训，而是“加载已有权重后开启一轮新实验”。这适合
    v3 这类 loss 权重小改动：保留 v2 已学到的几何校正能力，只观察新 loss
    是否能稳定校准裂缝区域位移幅度。
    """
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Initial checkpoint not found: {checkpoint_path}')

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt:
            ckpt = ckpt['model']

    cleaned = {}
    for key, value in ckpt.items():
        cleaned[key.replace('module.', '')] = value

    missing, unexpected = net.load_state_dict(cleaned, strict=False)
    log_message(f'Loaded initial checkpoint: {checkpoint_path}', log_file)
    if missing:
        log_message(f'  Missing keys: {len(missing)}', log_file)
    if unexpected:
        log_message(f'  Unexpected keys: {len(unexpected)}', log_file)


def log_message(msg, log_file=None):
    ts  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    out = f'[{ts}] {msg}'
    print(out)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(out + '\n')


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = getattr(config, 'benchmark', True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ──────────────────────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(net, val_loader, criterion, device, use_amp, log_file):
    net.eval()
    val_loss = val_epe = val_crack_epe = 0.0
    n = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        with autocast(device_type='cuda', enabled=use_amp):
            flows = net(images)
            loss, _ = criterion(flows if isinstance(flows, list) else [flows],
                                labels, img=images)
        val_loss += loss.item()
        final = flows[-1] if isinstance(flows, list) else flows
        val_epe += compute_epe(final, labels)
        if hasattr(criterion, 'crack_masker'):
            crack_mask = criterion.crack_masker(images).detach()
            diff = (final - labels) * 511.0
            epe_map = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True))
            crack_num = (epe_map * crack_mask).sum()
            crack_den = crack_mask.sum().clamp_min(1.0)
            val_crack_epe += (crack_num / crack_den).item()
        else:
            val_crack_epe += compute_epe(final, labels)
        n += 1
    return (
        val_loss / max(n, 1),
        val_epe / max(n, 1),
        val_crack_epe / max(n, 1),
    )


# ──────────────────────────────────────────────────────────────
# 主训练函数
# ──────────────────────────────────────────────────────────────

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    if config.restart_training and os.path.exists(config.output_dir):
        shutil.rmtree(config.output_dir, ignore_errors=True)
    os.makedirs(config.output_dir, exist_ok=True)

    log_file = os.path.join(config.output_dir, 'train.log')
    seed_everything(config.seed, deterministic=getattr(config, 'deterministic', False))

    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    log_message(f'Device: {device}, AMP: {use_amp}', log_file)

    # ── 数据集 ──
    log_message('Loading dataset...', log_file)
    full_train_set = ImageData(config.trainroot, training=True)
    full_val_set = ImageData(config.trainroot, training=False)
    total = len(full_train_set)
    if total == 0:
        raise RuntimeError(
            f'No training samples found under {config.trainroot}. '
            f'Expected *.png files with paired *.png.npy labels.'
        )
    val_size  = max(1, int(total * config.val_split))
    train_size = total - val_size
    train_idx_subset, val_idx_subset = Data.random_split(
        list(range(total)), [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    train_subset = Data.Subset(full_train_set, train_idx_subset.indices)
    val_subset = Data.Subset(full_val_set, val_idx_subset.indices)
    # 验证集关闭数据增强
    val_subset.dataset.augment.training = False
    log_message(f'Dataset: total={total}, train={train_size}, val={val_size}', log_file)

    log_message(
        f'Dataloader workers={config.workers}, pin_memory={torch.cuda.is_available()}',
        log_file
    )
    common_loader_kwargs = dict(
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=(config.workers > 0),
    )
    train_loader = Data.DataLoader(
        train_subset,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = Data.DataLoader(
        val_subset,
        batch_size=config.train_batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    # ── 模型 ──
    log_message('Building model...', log_file)
    net = build_crack_warp_net().to(device)
    load_initial_checkpoint(net, getattr(config, 'init_checkpoint', ''), device, log_file)
    # Ensure all parameters and buffers are on the correct device (robust move)
    for name, p in net.named_parameters():
        if p.device != device:
            p.data = p.data.to(device)
            if p.grad is not None:
                p.grad.data = p.grad.data.to(device)
    for name, b in net.named_buffers():
        if b.device != device:
            b.data = b.data.to(device)
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    log_message(f'Parameters: {total_params:.2f}M', log_file)

    # EMA（Mean Teacher 风格）
    ema = ModelEMA(net, decay=config.ema_decay) if config.use_ema else None

    # ── 损失 ──
    criterion = CrackWarpLoss(
        w_coord=config.w_coord,
        w_smooth=config.w_smooth,
        w_fold=config.w_fold,
        w_photo=config.w_photo,
        w_ssim=config.w_ssim,
        w_freq=config.w_freq,
        gamma=config.loss_gamma,
        charbonnier_eps=config.charbonnier_eps,
        w_crack_coord=getattr(config, 'w_crack_coord', 1.2),
        w_crack_grad=getattr(config, 'w_crack_grad', 0.25),
        w_crack_freq=getattr(config, 'w_crack_freq', 0.15),
        w_crack_mag=getattr(config, 'w_crack_mag', 0.0),
        crack_boost=getattr(config, 'crack_boost', 8.0),
        crack_topk=getattr(config, 'crack_topk', 0.08),
        crack_smooth_factor=getattr(config, 'crack_smooth_factor', 0.25),
    )
    # Move loss module buffers/params to device as well (SSIM window etc.)
    if hasattr(criterion, 'to'):
        criterion = criterion.to(device)

    # ── 优化器 ──
    # Transformer 权重使用更低 LR（参考 ViT fine-tuning 策略）
    transformer_params = []
    base_params        = []
    for name, p in net.named_parameters():
        if 'transformer' in name:
            transformer_params.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': base_params,        'lr': config.lr},
        {'params': transformer_params, 'lr': config.lr * 0.1},
    ], weight_decay=config.weight_decay)

    # ── 学习率调度：Warmup + CosineAnnealingWarmRestarts (SGDR) ──
    # 改进原因：原单周期余弦退火在epoch45后LR接近0，后35个epoch几乎无效
    # SGDR周期性重启LR，帮助模型跳出局部最优
    # 参考：'SGDR: Stochastic Gradient Descent with Warm Restarts' (ICLR 2017)
    # T_0=25: 第一个周期25epoch, T_mult=2: 后续周期加倍(25->50->100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        eta_min=config.min_lr,
    )
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    best_val_epe  = float('inf')
    best_val_crack_epe = float('inf')
    no_improve    = 0
    start_time    = time.time()

    log_message('Starting training...', log_file)
    log_message(f'Scheduler: CosineAnnealingWarmRestarts T_0=25 T_mult=2', log_file)
    log_message(
        f'Mixup: enabled from epoch {config.mixup_start_epoch}, '
        f'prob={config.mixup_prob}, alpha={config.mixup_alpha}',
        log_file
    )
    log_message(f'Checkpoint: saved every 5 epochs', log_file)
    log_message('=' * 80, log_file)

    for epoch in range(config.epochs):
        epoch_start = time.time()
        net.train()
        train_loss = train_epe = 0.0
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Mixup 增强（优化：推迟到epoch20，概率从50%降至30%，alpha从0.2降至0.1）──
            # 原因：坐标回归任务中过早/过强的Mixup会干扰精确映射学习
            # 分析：原配置Train/Val Loss差距4.4x，过强增强是主因之一
            if (epoch >= config.mixup_start_epoch and
                    torch.rand(1).item() < config.mixup_prob):
                images, labels = mixup_data(images, labels, alpha=config.mixup_alpha)

            with autocast(device_type='cuda', enabled=use_amp):
                flows = net(images)
                loss, loss_dict = criterion(flows, labels, img=images)
                loss = loss / config.accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % config.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(net)

            train_loss += loss_dict['total']
            final_flow  = flows[-1] if isinstance(flows, list) else flows
            train_epe  += compute_epe(final_flow.detach(), labels)

            if (i + 1) % config.display_interval == 0:
                log_message(
                    f'[{epoch+1}/{config.epochs}][{i+1}/{len(train_loader)}] '
                    f'loss={loss_dict["total"]:.5f} '
                    f'coord={loss_dict["coord"]:.5f} '
                    f'smooth={loss_dict["smooth"]:.5f} '
                    f'fold={loss_dict["fold"]:.5f} '
                    f'photo={loss_dict.get("photo", 0.0):.5f} '
                    f'freq={loss_dict["freq"]:.5f} '
                    f'c_coord={loss_dict.get("crack_coord", 0.0):.5f} '
                    f'c_grad={loss_dict.get("crack_grad", 0.0):.5f} '
                    f'c_freq={loss_dict.get("crack_freq", 0.0):.5f} '
                    f'c_mag={loss_dict.get("crack_mag", 0.0):.5f} '
                    f'c_ratio={loss_dict.get("crack_ratio", 0.0):.4f} '
                    f'lr={get_lr(optimizer):.2e}',
                    log_file
                )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_epe  = train_epe  / len(train_loader)

        # ── Epoch 级 LR 更新（Warmup + SGDR）──
        if epoch < config.warmup_epochs:
            # 线性 Warmup：从 min_lr 线性增至 config.lr
            warmup_lr    = config.min_lr + (config.lr - config.min_lr) * (epoch + 1) / config.warmup_epochs
            warmup_lr_tr = config.min_lr + (config.lr * 0.1 - config.min_lr) * (epoch + 1) / config.warmup_epochs
            optimizer.param_groups[0]['lr'] = warmup_lr
            optimizer.param_groups[1]['lr'] = warmup_lr_tr
        else:
            # SGDR：warmup结束后接管，会周期性重启LR
            scheduler.step(epoch - config.warmup_epochs)

        # ── 验证（使用 EMA 模型）──
        eval_net = ema.module if ema is not None else net
        avg_val_loss, avg_val_epe, avg_val_crack_epe = validate(
            eval_net, val_loader, criterion, device, use_amp, log_file)

        epoch_time = time.time() - epoch_start
        log_message(
            f'Epoch {epoch+1}/{config.epochs} | '
            f'Train Loss: {avg_train_loss:.5f} EPE: {avg_train_epe:.3f}px | '
            f'Val Loss: {avg_val_loss:.5f} EPE: {avg_val_epe:.3f}px '
            f'CrackEPE: {avg_val_crack_epe:.3f}px | '
            f'LR: {get_lr(optimizer):.2e} | Time: {epoch_time:.1f}s',
            log_file
        )

        # ── 保存最优模型 ──
        improved = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(config.output_dir, 'best_loss.pth')
            torch.save(eval_net.state_dict(), save_path)
            log_message(f'  Saved best loss model: {save_path}', log_file)
            improved = True

        if avg_val_epe < best_val_epe:
            best_val_epe = avg_val_epe
            save_path = os.path.join(config.output_dir, 'best_epe.pth')
            torch.save(eval_net.state_dict(), save_path)
            log_message(
                f'  Saved best EPE model (EPE={best_val_epe:.3f}px): {save_path}',
                log_file)
            improved = True

        if avg_val_crack_epe < best_val_crack_epe:
            best_val_crack_epe = avg_val_crack_epe
            save_path = os.path.join(config.output_dir, 'best_crack_epe.pth')
            torch.save(eval_net.state_dict(), save_path)
            log_message(
                f'  Saved best crack-EPE model (CrackEPE={best_val_crack_epe:.3f}px): {save_path}',
                log_file)
            improved = True

        # ── 早停（SGDR下：LR重启时重置计数，避免重启瞬间误触发早停）──
        # 检测LR重启：scheduler内部epoch接近T_0的倍数时LR会回到峰值
        current_lr = get_lr(optimizer)
        lr_restarted = (epoch >= config.warmup_epochs and
                        current_lr > config.lr * 0.8)
        if lr_restarted:
            # LR刚重启，重置早停计数，给模型新周期的机会
            if no_improve > 0:
                log_message(f'  LR restarted, reset early stop counter', log_file)
            no_improve = 0
        elif improved:
            no_improve = 0
        else:
            no_improve += 1
            log_message(
                f'  No improvement for {no_improve}/{config.early_stop_patience} epochs',
                log_file)
            if no_improve >= config.early_stop_patience:
                log_message(
                    f'Early stopping triggered at epoch {epoch+1}', log_file)
                break

        # ── 周期性检查点（每5epoch保存，防止服务器重启丢失进度）──
        # 原：每10epoch保存一次，服务器重启最多丢失10epoch进度
        # 改：每5epoch保存一次，最多丢失5epoch进度
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(config.output_dir, f'epoch_{epoch+1}.pth')
            torch.save(eval_net.state_dict(), ckpt_path)
            log_message(f'  Saved checkpoint: {ckpt_path}', log_file)

    # ── 训练完成 ──
    total_time = time.time() - start_time
    final_path = os.path.join(config.output_dir, 'final.pth')
    eval_net = ema.module if ema is not None else net
    torch.save(eval_net.state_dict(), final_path)

    log_message('=' * 80, log_file)
    log_message('Training completed!', log_file)
    log_message(f'Total time: {total_time/3600:.2f} hours ({total_time:.0f}s)', log_file)
    log_message(f'Best Val Loss: {best_val_loss:.5f}', log_file)
    log_message(f'Best Val EPE:  {best_val_epe:.3f}px', log_file)
    log_message(f'Best Crack EPE: {best_val_crack_epe:.3f}px', log_file)
    log_message(f'Final model:   {final_path}', log_file)
    log_message(f'Log file:      {log_file}', log_file)


if __name__ == '__main__':
    main() 
