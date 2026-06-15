# 最终全量主模型汇总

说明：下表基于 10360 张全量数据评估结果。箭头表示指标方向；EPE 和 folding 越低越好，Dice 和 edge fidelity 越高越好。

| 实验 | 角色 | Crack EPE↓ | Global EPE↓ | Dice↑ | Crack Edge↑ | Global Edge↑ | Folding↓ | v4更优/对方更优 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4_robust_edge_10ep_all | 全量：EPE/Dice 保守主模型 | 114.416328 | 111.964767 | 0.276011 | 0.111250 | 0.098206 | 0.583643 | - | EPE/Dice 略优，保守主模型 |
| v5_jacobian_roi_w002_10ep_lr1e6_all | 全量：folding 最优几何稳定候选 | 114.440132 | 112.246910 | 0.274965 | 0.120241 | 0.109083 | 0.579404 | 4857/5503 | folding 最优，综合评分优于 v4 |
| v6_recover_epe_from_v5_10ep_all | 全量：edge 增强补充实验 | 114.494644 | 112.070282 | 0.274207 | 0.124766 | 0.114296 | 0.582038 | 4650/5710 | edge 最优，综合评分优于 v4，但 EPE/Dice 有代价 |

## 结论

- 全量口径下 `v4_robust_edge_10ep` 的 EPE/Dice 均值更稳，`v5_jacobian_roi_w002_10ep_lr1e6` 的 folding 最优，`v6_recover_epe_from_v5_10ep` 的 edge fidelity 最优。
- 全量逐图综合评分中，v5 相对 v4 为 5503/10360 更优，v6 相对 v4 为 5710/10360 更优；因此 v5/v6 在综合覆盖面上优于 v4，但 v4 仍保留 EPE/Dice 均值优势。
- 4 个 2 epoch 消融均未超过完整 v4，说明鲁棒位移、过大位移惩罚和边缘一致性组合具有必要性。
- `unimatch_oracle_pair` 和 `searaft_oracle_pair` 使用 GT 校正图和输入图做 dense matching，属于 oracle-pair 上界参考，不能作为同输入条件方法直接压过主模型来表述。
- 后续若继续做模型优化，应优先设计 folding/Jacobian 正则、边界平滑或 ROI 局部对齐消融，而不是简单拉长这 4 个配置。
