# GOOD-CNN CrackWarp 优化指南（2026-04）

## 1. 先重建高质量训练集

先用改进后的逆映射生成器重建数据（`compute_inverse_map` 改为 fixed-point 迭代）：

```bash
python generate_underwater_v2.py
```

生成后建议至少做两项检查：

1. 随机抽样 `distorted + label.npy`，用 `cv2.remap` 逆映射看是否回到原图几何。
2. 检查标签范围是否稳定在 `[0, 1]`，并统计越界比例应接近 0。

## 2. 训练端关键改动（已落地）

当前训练逻辑已做以下修复：

1. 训练/验证集改为独立 Dataset，避免增强策略串扰。
2. DataLoader 改为可配置 workers + `persistent_workers` + worker 随机种子。
3. 新增 photometric consistency loss（Pred-warp vs GT-warp）与 SSIM 项。
4. Mixup 改为更保守课程学习（可在配置调节起始 epoch、概率、alpha）。
5. 数据集读取时强制过滤缺失/损坏 `.npy` 标签，避免“静默坏样本”。

## 3. 推荐训练命令

```bash
python train_v2.py
```

关键配置位于 `config_crack.py`：

1. `w_photo=0.15, w_ssim=0.10`：提升视觉复原一致性。
2. `mixup_start_epoch=30, mixup_prob=0.10, mixup_alpha=0.05`：减轻几何回归被过强 mixup 干扰。
3. `workers=8`：提升吞吐，保证每轮更多有效更新。

## 4. 推理命令

```bash
python infer_epoch80.py --model output_crackwarp/best_epe.pth --img_dir <你的测试目录> --out_dir output_crackwarp/infer_results --num -1
```

`infer_epoch80.py` 已支持 `png/jpg/jpeg`。

## 5. 自动评估（第二轮冲指标）

```bash
python utils/evaluate_metrics.py --model output_crackwarp/best_epe.pth --img_dir underwater_crack_v3 --out_dir output_crackwarp/eval_metrics --num -1
```

输出：

1. `eval_summary.json`：全局均值/方差（EPE、边缘保真、折叠率）
2. `eval_per_image.csv`：逐图指标，便于筛坏样本

## 6. 日志驱动调参（第二轮冲指标）

```bash
python utils/tune_from_log.py --log output_crackwarp/train.log --out output_crackwarp/tuning_plan.json
```

输出：

1. 训练诊断（underfit/overfit/plateau）
2. 可直接试跑的 `suggested_trials` 参数组合

## 7. 对齐的论文方向（建议继续迭代）

1. RAFT（ECCV 2020）：多阶段流监督与迭代更新思想  
   https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf
2. Restormer（CVPR 2022）：高分辨率恢复的高效 transformer 思路  
   https://arxiv.org/abs/2111.09881
3. FlowFormer（CVPR 2022）/ FlowFormer++（CVPR 2023）：更强全局流建模  
   https://arxiv.org/abs/2203.16194  
   https://openaccess.thecvf.com/content/CVPR2023/html/Shi_FlowFormer_Transformer_for_Optical_Flow_CVPR_2023_paper.html
4. DocReal（WACV 2024）：真实场景文档去畸变难例与鲁棒性  
   https://openaccess.thecvf.com/content/WACV2024/html/Yu_DocReal_Robust_Document_Dewarping_of_Real-Life_Images_via_Attention-Enhanced_Control_WACV_2024_paper.html

---

如果下一步要继续冲效果，优先级建议是：  
`真实域数据增强` > `标签精度` > `损失权重微调` > `模型结构加大`。
