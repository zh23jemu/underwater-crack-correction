# -*- coding: utf-8 -*-
"""Training config for CrackWarpNet."""

# ---- Data ----
trainroot = './underwater_crack_v3/'
output_dir = './output_crackwarp/'
shape = (512, 512)

# ---- Training ----
gpu_id = 0
workers = 8
seed = 42
start_epoch = 0
epochs = 50  # first run 50 epochs for quick iteration
train_batch_size = 1
val_split = 0.1
deterministic = False
benchmark = True

# ---- Learning rate ----
lr = 2e-5
warmup_epochs = 5
min_lr = 1e-7

# ---- Regularization ----
weight_decay = 5e-4
gradient_clip = 0.5
dropout = 0.1

# ---- Loss weights ----
w_coord = 1.0
w_smooth = 0.1
w_fold = 0.01
w_photo = 0.15
w_ssim = 0.10
w_freq = 0.10
charbonnier_eps = 1e-3

# ---- Crack-centric loss weights ----
w_crack_coord = 1.20
w_crack_grad = 0.25
w_crack_freq = 0.15
w_crack_mag = 0.0  # 位移幅度一致性损失，默认关闭；v3 诊断训练可设置为 0.2-0.5
crack_boost = 8.0
crack_topk = 0.08
crack_smooth_factor = 0.25

# ---- Multi-scale supervision ----
loss_gamma = 0.8

# ---- EMA ----
use_ema = True
ema_decay = 0.999

# ---- Early stop ----
early_stop_patience = 20

# ---- Gradient accumulation ----
accum_steps = 4

# ---- Data curriculum ----
mixup_start_epoch = 30
mixup_prob = 0.10
mixup_alpha = 0.05

# ---- Logging / eval ----
display_interval = 20
eval_interval = 1
vis_interval = 10

# ---- Resume ----
restart_training = True
checkpoint = ''
init_checkpoint = ''  # 仅加载模型权重做 fine-tune，不恢复 optimizer/scheduler 状态
