# CrackWarp 项目逻辑与全流程使用说明

## 1. 项目目标
本项目用于水下裂缝图像的几何扭曲复原。模型预测归一化逆映射坐标场 `flow (2, H, W)`，再通过 `cv2.remap` 进行矫正。

核心原则：优先保证裂缝结构（细线、分支、连续性），而不是只追求全图平均相似度。

## 2. 核心文件与职责
- `generate_underwater_v2.py`：合成训练数据（扭曲图 + `.npy` 逆映射标签）
- `MyDataSet.py`：数据集读取与增强（几何增强会同步作用在图像和 flow 标签）
- `models/crack_warp_net.py`：主网络（编码器-解码器 + Transformer + 迭代细化）
- `loss_crack.py`：裂缝优先损失（含 crack attention 与 crack 区域损失项）
- `config_crack.py`：训练配置参数
- `train_v2.py`：训练入口（训练/验证/EMA/保存权重）
- `utils/evaluate_metrics.py`：裂缝中心评估指标
- `infer_epoch80.py`：推理与可视化（脚本名沿用历史命名）

## 3. 数据格式
每个样本成对出现：
- 图像：`xxx.png`
- 标签：`xxx.png.npy`

标签格式：
- `shape = (2, H, W)`
- `label[0]` 为归一化 `x` 坐标，`[0,1]`
- `label[1]` 为归一化 `y` 坐标，`[0,1]`

像素坐标恢复：
- `map_x = label[0] * (W - 1)`
- `map_y = label[1] * (H - 1)`

## 4. 训练逻辑
1. 输入扭曲图像
2. 网络输出多阶段 flow：`[coarse, iter1, ..., iter_n]`
3. `CrackWarpLoss` 计算总损失（全局 + 裂缝区域）
4. 反向传播更新
5. 每个 epoch 验证并保存 `best_loss.pth / best_epe.pth / epoch_xx.pth / final.pth`

## 5. 从头到尾使用步骤

### Step 0：准备环境
建议 Python 3.9+，安装依赖：
- `torch`, `torchvision`
- `opencv-python`
- `numpy`, `scipy`, `tqdm`
- `Pillow`, `natsort`, `scikit-image`

### Step 1：准备原始裂缝图
将原始图像放到：
- `./under-crack-images/images/`

### Step 2：生成训练数据
```bash
python generate_underwater_v2.py
```
默认输出目录：`./underwater_crack_v3/`

### Step 3：检查训练配置
编辑 `config_crack.py`，重点看：
- `trainroot`, `output_dir`
- `epochs`, `train_batch_size`, `lr`
- 裂缝损失相关参数（`w_crack_*`, `crack_*`）

当前配置已改为：
- `epochs = 50`（先跑 50 epoch 做第一轮验证）

### Step 4：启动训练
```bash
python train_v2.py
```
训练日志：`output_crackwarp/train.log`

### Step 5：评估（裂缝中心指标）
```bash
python utils/evaluate_metrics.py --model output_crackwarp/best_epe.pth --img_dir underwater_crack_v3 --out_dir output_crackwarp/eval_metrics
```
重点看：
- `primary_crack_epe_px_mean`
- `primary_crack_edge_fidelity_mean`
- `primary_warp_crack_dice_mean`

### Step 6：推理与可视化
```bash
python infer_epoch80.py --model output_crackwarp/best_epe.pth --img_dir underwater_crack_v3 --out_dir output_crackwarp/infer_results --num 20
```
输出：`*_corrected.png`, `*_compare.png`, 汇总图。

## 6. 推荐复现顺序
```bash
python generate_underwater_v2.py
python train_v2.py
python utils/evaluate_metrics.py --model output_crackwarp/best_epe.pth --img_dir underwater_crack_v3 --out_dir output_crackwarp/eval_metrics
python infer_epoch80.py --model output_crackwarp/best_epe.pth --img_dir underwater_crack_v3 --out_dir output_crackwarp/infer_results --num 20
```

## 7. 常见问题
- 找不到样本：检查 `trainroot` 与 `.png/.png.npy` 是否配对
- 裂缝发糊：提高 `w_crack_coord / w_crack_grad / w_crack_freq`，适当降低 `w_smooth`
- 结果有撕裂：关注 `folding_rate`，可适当提高 `w_fold`
