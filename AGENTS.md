# 项目级 AGENTS.md

## 项目目标

本项目用于整理、分析、训练和后续优化“水下裂缝图像扭曲校正”算法。当前仓库已经包含用户补充的代码、合成数据、原始裂缝图、checkpoint、训练日志、评估结果和两份 Word 报告。核心任务是提升水下裂缝扭曲复原效果，重点解决当前可视化效果不好、裂缝复原不足、预测恢复量偏小、工程门限未达标等问题。

## 技术栈

- 文档格式：Microsoft Word `.docx`
- 报告生成痕迹：文档元数据均显示由 `python-docx` 生成
- 运行环境：已创建项目本地 Python 虚拟环境 `.venv`；当前 `.venv` 已安装文档处理依赖和 `numpy`，但尚未安装训练所需的 PyTorch、OpenCV、SciPy、tqdm、natsort、scikit-image 等依赖
- 深度学习框架：PyTorch 代码结构，训练入口为 `train_v2.py`
- 图像与数据：OpenCV / NumPy / Pillow 风格的数据生成、读取、推理和可视化流程
- 已新增 Slurm 训练包装入口和提交脚本；集群训练应优先使用 `slurm_train_crackwarp.sbatch`，通过 `run_train_slurm.py` 覆盖输出目录和训练参数，避免误清空历史结果

## 当前架构

- `水下裂缝扭曲校正实验报告(1).docx`：普通版报告，侧重训练收敛、典型样本可视化和综合结论。
- `水下裂缝扭曲校正实验报告_详细版(1).docx`：详细版报告，额外包含 checkpoint 横向评估、逐图像误差分布、ROI 局部观察、门限失败原因和优化建议。
- `generate_underwater_v2.py`：根据真实裂缝图合成水下扭曲训练样本与 `.npy` 逆映射标签。
- `MyDataSet.py`：读取成对的 `.png` / `.png.npy` 样本，并实现水下颜色、散射、噪声、模糊和几何增强。
- `models/crack_warp_net.py`：主网络，包含编码器-解码器、Transformer、CBAM、NAFBlock 和迭代 flow refinement 等结构。
- `loss_crack.py`：裂缝优先损失，包含坐标损失、边缘感知平滑、folding penalty、频域损失、光度一致性和 crack mask 加权项。
- `train_v2.py`：训练入口，使用 `config_crack.py` 配置，包含训练/验证划分、AMP、EMA、warm restart、mixup、checkpoint 保存等逻辑。
- `infer_epoch.py`：推理与可视化入口；现有文档中仍有历史脚本名 `infer_epoch80.py`，后续需要统一。
- `utils/evaluate_metrics.py` 与 `utils/checkpoint_gate_report.py`：裂缝中心指标评估和 checkpoint 门限诊断。
- `utils/compare_eval_per_image.py`：对比新旧模型 `eval_per_image.csv`，自动挑选新模型更好、旧模型更好和共同失败的诊断样本，并输出图片清单。
- `utils/export_crack_roi_visuals.py`：裂缝 ROI 可视化导出脚本，现已支持 `--image_list`，可按诊断样本清单批量出局部放大图。
- `utils/export_flow_diagnostics.py`：整图 flow 诊断脚本，用于同批样本的新旧模型对比，导出校正图、EPE 热图、folding 热区、位移幅度图和逐图诊断 CSV。
- `under-crack-images/images/`：原始真实裂缝图，共 1,037 张 `.jpg`。
- `underwater_crack_v3/`：主合成训练集，共 10,360 张 `.png` 与 10,360 个 `.png.npy` 标签，`manifest.json` 记录 `samples_per_image=10` 与 `total_generated=10360`。
- `output_crackwarp/`：已有训练输出，包含 `train.log`、13 个 `.pth` checkpoint、评估 CSV/JSON、gate report 和可视化结果。
- `.gitignore`：忽略本地虚拟环境、Python 缓存、编辑器目录和系统生成文件。
- `.venv/`：项目本地 Python 虚拟环境，不应入库。

## 开发规范

- 所有 Python 操作必须使用项目本地 `.venv`，不要使用系统 Python。
- 编辑文件前必须先读取和理解现有内容。
- 保持最小修改，不做无关重构。
- 不直接删除文件；如需清理文件，只提供建议命令或先征求确认。
- 新增代码或脚本时，应包含较详细的中文注释，说明用途、关键逻辑、参数、返回值和重要分支。
- 后续若新增训练代码、推理脚本或长时间运行任务，应评估是否需要 Slurm 脚本；GPU 任务默认优先使用 `gpu` 分区，账号默认 `gpo-ifv7xx`，QOS 默认 `normal`。
- 当前 `train_v2.py` 中 `restart_training=True` 会删除目标输出目录；Slurm 训练必须使用独立输出目录，默认写入 `output_crackwarp_slurm/<jobid>`，不要直接覆盖已有 `output_crackwarp/`。

## Current Status

已完成用户补充代码和数据后的只读确认。当前仓库已经从纯报告资料仓库变为可继续工程诊断和训练优化的算法项目：总计约 21,913 个非 `.venv` 文件、约 27.25GB，其中主训练集 `underwater_crack_v3` 约 26.21GB，训练输出 `output_crackwarp` 约 0.98GB。主数据集配对完整，抽样标签为 `(2,512,512)` 的 `float32`，数值范围稳定在 `[0,1]`。已有训练跑满 50 epoch，最佳 Val EPE 约 117.885px；gate report 显示所有 checkpoint 均为 `below_minimum`，最佳 `best_epe.pth` 的 crack EPE 约 112.021px、Dice 约 0.0026、edge fidelity 约 0.295、global EPE 约 112.606px、folding rate 约 0.570，距离工程门限仍有明显差距。已新增 Slurm 训练入口，但本机 `.venv` 是从旧机器拷贝来的失效环境，当前无法执行 `.venv\Scripts\python.exe`，需要在 Slurm/新机器上重建虚拟环境后再训练。

已完成项目下全部 Markdown 与 Word 文档的只读阅读和目标归纳：文档总体确认本项目的核心目标不是普通图像增强，而是学习水下裂缝扭曲图像的归一化逆映射坐标场，通过 `cv2.remap` 将扭曲裂缝恢复到更接近无扭曲的几何形态。项目评价重点应优先围绕裂缝主体位置、细裂缝分叉、边缘锐度、局部 ROI 对齐、预测恢复幅度和位移场可逆性，而不是只看全图平均损失。普通版实验报告强调模型已稳定收敛并具备一定全局校正能力；详细版报告、gate report、交接文档和工作量说明则共同指出当前结果仍未达到工程或论文级验收标准，下一阶段应以复现诊断、坐标/逆映射一致性排查、Slurm 全量训练、裂缝区域监督增强、folding/Jacobian 约束、ROI 局部细化、对比实验和消融实验为主线。

截至 2026-06-07，本地目录已同步为轻量代码仓库版本：大数据目录 `underwater_crack_v3/`、`under-crack-images/` 和历史输出 `output_crackwarp/` 不在当前本地工作区内，已通过 GitHub Release `data-v1` 分发，并据 `新电脑接续工作指南.md` 记录已在服务器 `usmidet-com-prod-gpu001` 下载、校验、合并和解压完成。当前本地重点文件包括 `requirements.txt`、`run_train_slurm.py`、`slurm_train_crackwarp.sbatch`、`smoke_train_verify.py`、`GITHUB_UPLOAD_PLAN.md`、`SHA256SUMS.txt` 和 `新电脑接续工作指南.md`。下一步主要工作应在服务器上确认数据数量、重建 `.venv`、跑 Slurm smoke，然后再提交正式训练。

截至 2026-06-08，服务器 Slurm 正式 50 epoch 训练已完成。任务 `34827388` 使用 `EPOCHS=50 OUTPUT_DIR=output_crackwarp_slurm/v2 sbatch --partition=gpu slurm_train_crackwarp.sbatch` 提交到 `gpu` 分区，并在 `usmidet-com-prod-gpu001` 正常跑完；`slurm-crackwarp-train-34827388.err` 为空，说明本轮未出现 OOM 或 Python Traceback。输出目录 `output_crackwarp_slurm/v2` 约 964MB，包含 `best_loss.pth`、`best_epe.pth`、`best_crack_epe.pth`、`epoch_5.pth` 至 `epoch_50.pth`、`final.pth`、`train.log` 和 `eval_best_epe_120/`。自动 120 样本评估显示：crack EPE 108.987938、global EPE 111.060646、crack Dice 0.283552、crack edge fidelity 0.308878、global edge fidelity 0.294752、folding rate 0.590619。相比历史 `output_crackwarp/best_epe.pth` 的 120 样本复评结果，EPE 和 Dice 有小幅提升，但 edge fidelity 略低、folding rate 更高，整体仍未达到三区论文目标或工程门限，下一阶段应从坐标/逆映射逻辑、folding 约束、裂缝 ROI 监督和可视化诊断继续优化。

2026-06-08 继续补评 `output_crackwarp_slurm/v2/best_crack_epe.pth` 与 `best_loss.pth`，两者在 120 样本上的评估结果与 `best_epe.pth` 完全一致，说明本轮训练中三个 best 权重很可能对应同一保存时刻或同一 epoch。随后对 `best_epe.pth` 扩大到 1000 样本评估，结果为 crack EPE 113.286263、global EPE 113.095718、crack Dice 0.255906、crack edge fidelity 0.204312、global edge fidelity 0.224700、folding rate 0.590347。该结果比 120 样本更差，说明 120 样本评估偏乐观；后续报告和模型判断应优先采用更大样本或固定验证集指标。

2026-06-08 进一步用同一 1000 样本口径复评历史 `output_crackwarp/best_epe.pth`，结果为 crack EPE 116.231331、global EPE 114.404533、crack Dice 0.252041、crack edge fidelity 0.218584、global edge fidelity 0.242924、folding rate 0.569224。与 v2 相比，v2 的 crack/global EPE 和 Dice 小幅更好，但 edge fidelity 与 folding rate 更差。因此 v2 的真实结论是“位置误差略有改善，但裂缝边缘复原和位移场可逆性没有改善”，不适合作为最终达标模型。

2026-06-08 已从服务器拉取 `f6f4645` 诊断结果，包含 `output_crackwarp_slurm/v2/compare_old_vs_v2/` 和共同失败样本的新旧 ROI 图。`compare_summary.json` 显示 1000 张匹配样本中 v2 综合分更好的有 523 张，旧模型更好的有 477 张，说明 v2 不是稳定全面提升，而是接近五五开的局部改善。共同失败样本高度集中在 `crack0030_xx` 一组，ROI 图显示大形变和边界附近存在明显局部涂抹、重复纹理、不自然边缘和裂缝主体偏移；例如 `crack0030_08`、`crack0030_02`、`crack0063_05` 中 v2 虽然部分纹理位置有变化，但裂缝细节没有稳定复原，部分样本反而抹淡或拉偏。该诊断进一步支持下一步应先修正/增强坐标约束、folding/Jacobian 约束、边缘保持和 ROI 局部监督，而不是继续同配置堆训练轮数。

2026-06-08 新增 `utils/export_flow_diagnostics.py`，用于服务器下一步继续诊断共同失败样本。该脚本同时加载旧模型和 v2 新模型，对指定图片清单导出整图诊断面板，包含输入、旧模型校正、v2 校正、GT 校正、旧/新 EPE 热图、旧/新 folding 热区和旧/新位移幅度图，并生成 `flow_diagnostics.csv`。该脚本需要在服务器 `.venv/bin/python` 下做语法检查和实际运行，本地 Windows 工作区无 `.venv`，暂不能本地执行验证。

2026-06-08 已拉取服务器提交 `d87f9bb`，包含 `output_crackwarp_slurm/v2/flow_diag_joint_failures/` 的 20 张共同失败样本整图诊断面板和 `flow_diagnostics.csv`。CSV 显示这 20 张样本中 v2 全局 EPE 平均从 129.430 降到 123.972，但裂缝区域 EPE 平均从 176.045 升到 177.491；v2 平均位移幅度从 128.289px 降到 121.280px，说明 v2 更偏保守/平滑，全图指标可能改善，但裂缝主体并未稳定对准。典型样本 `crack0041_07` 中 v2 明显优于旧模型，主要因为旧模型位移幅度过大，而 v2 降低位移后 EPE 大幅下降；典型样本 `crack0063_05` 中 v2 严重劣化，全局 EPE 从 114.437 升到 182.805，裂缝 EPE 从 114.093 升到 208.985，且 v2 位移幅度从 115.780px 增到 182.945px，说明 v2 在部分样本上会产生过强或方向错误的局部恢复。当前判断：问题不是单纯“恢复量太小”，而是恢复幅度缺少结构约束，有时过小、有时过大，且没有稳定服务于裂缝 ROI。

2026-06-08 拉取服务器正反样本诊断结果后，确认 v2 改善/退化与位移幅度强相关：`flow_diag_new_better` 的 20 张样本中，v2 裂缝 EPE 全部改善，平均位移幅度从 163.353px 降到 116.598px；`flow_diag_old_better` 的 20 张样本中，v2 裂缝 EPE 全部变差，平均位移幅度从 123.072px 升到 144.373px。由此将下一轮 v3 试验定义为“位移幅度一致性校准”，而不是简单增大或减小恢复量。

2026-06-08 已在本地新增 `DisplacementMagnitudeConsistencyLoss`，并通过 `w_crack_mag` 参数接入 `CrackWarpLoss`、`train_v2.py`、`run_train_slurm.py` 和 `slurm_train_crackwarp.sbatch`。该损失默认权重为 0，不影响当前 v2 复现；服务器可通过环境变量 `W_CRACK_MAG=0.3` 启用，用于约束裂缝 ROI 内预测位移幅度接近 GT 位移幅度，降低样本间过小/过大的恢复幅度漂移。

2026-06-08 服务器运行 `W_CRACK_MAG=0.3 EPOCHS=2 OUTPUT_DIR=output_crackwarp_slurm/v3_mag_smoke sbatch --partition=gpuHz --time=01:00:00 slurm_train_crackwarp.sbatch` 已完成，`.err` 为空但指标灾难性：global EPE 约 298.617、folding rate 约 0.99938。该结果说明新 loss 链路能跑通，但从随机初始化训练 2 epoch 没有诊断价值，且可能把早期位移场推向严重折叠。下一步已改为支持从 v2 `best_epe.pth` 初始化进行短程 fine-tune。

2026-06-08 本地新增 `INIT_CHECKPOINT` / `--init-checkpoint` 入口，训练时只加载模型权重，不恢复 optimizer/scheduler/scaler。该入口用于 v3 fine-tune：从 `output_crackwarp_slurm/v2/best_epe.pth` 初始化，以较小学习率和较低 `W_CRACK_MAG` 验证位移幅度一致性损失是否能在不破坏 v2 几何能力的前提下改善裂缝 ROI 稳定性。

2026-06-09 服务器已完成 v3 fine-tune smoke：`INIT_CHECKPOINT=output_crackwarp_slurm/v2/best_epe.pth W_CRACK_MAG=0.1 LR=5e-6 EPOCHS=2 OUTPUT_DIR=output_crackwarp_slurm/v3_mag_ft_smoke sbatch --partition=gpuHz --time=02:00:00 slurm_train_crackwarp.sbatch`。日志确认已加载 v2 checkpoint，`c_mag` 正常出现，`.err` 为空。120 样本评估结果为 crack EPE 107.956345、crack edge fidelity 0.306520、Dice 0.284974、global EPE 109.795090、global edge fidelity 0.293271、folding rate 0.589715。相较 v2 的 120 样本结果，crack/global EPE 与 Dice 小幅改善，folding 略好，edge 基本持平略低，说明从 v2 初始化后开启 `w_crack_mag=0.1` 是可继续验证的方向；下一步应做 1000 样本评估和 v2-v3 逐图对比，不能直接扩大到 50 epoch。

2026-06-09 已拉取服务器提交 `63a0971`，包含 v3 fine-tune 的 1000 样本评估和 v2-v3 逐图对比结果。v3 1000 样本结果为 crack EPE 112.940239、global EPE 111.840080、Dice 0.259461、crack edge fidelity 0.200950、global edge fidelity 0.221410、folding rate 0.589862。相比 v2 1000 样本结果，v3 的 crack EPE 改善约 0.346px，global EPE 改善约 1.256px，Dice 提升约 0.00355，folding rate 改善约 0.00048，但 crack/global edge fidelity 分别下降约 0.00336/0.00329。逐图综合评分显示 1000 张中 v3 优于 v2 的有 644 张，劣于 v2 的有 356 张，平均综合分提升 1.017554。当前结论：`w_crack_mag=0.1` 微调方向有效，能更稳定地校准位移幅度，但边缘保持仍未改善，下一步应先做 v2-v3 正反样本 flow 诊断，再考虑 10 epoch 小规模 fine-tune，而不是直接长训 50 epoch。

2026-06-09 服务器已完成 v2-v3 正反样本 flow 诊断导出：`flow_diag_new_better` 与 `flow_diag_old_better` 各处理 20 张样本，控制台显示 new-better 组大多从 v2 到 v3 稳定降低 global EPE，例如 `crack0038_01` 从 107.428 降到 104.631；old-better 组中多个高难样本进一步变差，例如 `crack0053_03` 从 151.745 升到 159.044、`crack0063_05` 从 182.805 升到 188.962、`crack0076_04` 从 200.138 升到 206.474。下一步需要读取两组 `flow_diagnostics.csv` 的位移幅度、crack EPE 和 folding 均值，确认 v3 变差样本是否来自位移幅度继续偏大、方向错误或边界 folding。

2026-06-09 已拉取服务器提交 `5b9ca78` 并分析 v2-v3 flow 诊断 CSV/图片。`flow_diag_new_better` 20 张中，v3 相比 v2 的 global EPE 平均从 112.420 降到 110.474，crack EPE 从 116.845 降到 113.667，mean displacement magnitude 从 109.689px 降到 107.996px，p95 displacement magnitude 从 214.854px 降到 212.139px，folding rate 从 0.392338 小升到 0.393053。`flow_diag_old_better` 20 张中，v3 的 global EPE 从 136.226 升到 139.193，crack EPE 从 126.328 升到 134.865，mean displacement magnitude 从 136.213px 升到 139.161px，p95 displacement magnitude 从 247.489px 升到 251.113px，folding rate 从 0.398417 升到 0.400513。代表性图片 `crack0038_01` 显示 v3 通过收缩位移幅度降低 EPE；`crack0053_03`、`crack0079_09`、`crack0100_09` 显示退化样本主要是右侧/边界大形变区域位移幅度进一步变强，导致 EPE 热区更重。当前判断：v3 的 `w_crack_mag=0.1` 不是稳定边缘复原方案，而是一个位移幅度校准信号；它对“v2 位移偏大”的样本有效，但仍会把部分高难样本推向更大恢复幅度，后续应先做更保守的权重或位移幅度上限/鲁棒化约束。

2026-06-09 已拉取服务器提交 `3346ffd`，包含 `W_CRACK_MAG=0.05`、`LR=5e-6`、从 v2 初始化微调 10 epoch 的评估结果。该版本 `output_crackwarp_slurm/v3_mag_ft_w005_10ep/best_epe.pth` 在 1000 样本上达到 crack EPE 111.806763、global EPE 110.886253、Dice 0.260308、crack edge fidelity 0.202303、global edge fidelity 0.224912、folding rate 0.586779。相比 v2，crack EPE 改善约 1.480px、global EPE 改善约 2.209px、Dice 提升约 0.0044、global edge 基本持平略好、folding 改善约 0.00357；相比 `W_CRACK_MAG=0.1` 的 2 epoch smoke，也进一步改善 EPE、Dice、global edge 和 folding。逐图综合评分显示 1000 张中 850 张优于 v2、150 张劣于 v2，global EPE 有 918 张改善，crack EPE 有 802 张改善，folding 有 854 张改善。当前结论：`W_CRACK_MAG=0.05 + 10 epoch` 是目前最稳定的主线候选模型，但 crack edge fidelity 仍略低于 v2，且退化样本集中在 `crack0063`、`crack0079`、`crack0053`、`crack0052` 等高难/大形变组。

2026-06-09 已拉取服务器提交 `c200b54`，包含 `v3_mag_ft_w005_10ep` 的退化样本和共同失败样本 flow 诊断图/CSV。`flow_diag_old_better` 20 张退化样本中，v3 相比 v2 的 global EPE 从 141.950 升到 147.531，crack EPE 从 139.169 升到 152.091，folding rate 从 0.401218 升到 0.405723，mean displacement magnitude 从 142.239px 升到 147.605px，p95 displacement magnitude 从 253.089px 升到 260.220px；其中 crack EPE 没有任何一张改善。`flow_diag_joint_failures` 20 张中，v3 global EPE 从 133.049 小降到 132.855，15 张 global EPE 改善，但 crack EPE 从 179.567 升到 181.797，folding rate 从 0.395344 升到 0.400381，说明 w005_10ep 对整图位置有帮助，但裂缝 ROI 仍有局部过强/折叠和边缘对齐问题。代表图 `crack0079_09`、`crack0063_07` 显示退化主要发生在边缘和大形变区域，v3 位移热图更强、folding 斑点更密；`crack0030_03` 说明共同失败族整体有改善但裂缝细节仍未完全对齐。当前判断：下一阶段应从“继续调低 `W_CRACK_MAG` 或训练更久”转向加入位移幅度上限/鲁棒化、边界约束和裂缝边缘一致性约束。

2026-06-09 已完成下一步代码修改：`DisplacementMagnitudeConsistencyLoss` 新增 `crack_mag_robust_delta` 和 `crack_mag_over_weight`，用于对位移幅度一致性做 Huber 式鲁棒化，并额外惩罚预测位移幅度超过 GT 的部分，降低高难样本 p95 displacement magnitude 被继续推大的风险；新增 `WarpedImageGradientConsistencyLoss`，通过预测校正图与 GT 校正图的灰度梯度差异，在 GT 校正后的裂缝 ROI 附近约束边缘结构。新增参数默认均为 0，保持历史配置不变；Slurm 包装入口已支持 `W_CRACK_EDGE`、`CRACK_MAG_ROBUST_DELTA`、`CRACK_MAG_OVER_WEIGHT` 环境变量。下一步需在服务器 `.venv/bin/python` 下做 `py_compile` 和 2 epoch smoke，本地 Windows 工作区无 `.venv`，未进行 Python 编译验证。

## Recent Changes

- 新增 `.gitignore`，忽略 `.venv/`、Python 缓存、系统文件和常见编辑器目录。
- 新建项目本地 `.venv`，用于满足本项目 Python 环境约束。
- 新增本项目级 `AGENTS.md`，记录项目目标、当前结构、维护规范、进展、风险和下一步计划。
- 只读解析了两份用户提供的 `.docx` 报告，未修改报告原文。
- 新增 `水下裂缝扭曲校正项目工作内容与工作量说明.md`，作为对外报价和项目排期说明的可编辑源稿。
- 新增 `水下裂缝扭曲校正项目工作内容与工作量说明.docx`，作为可直接发给客户/甲方的 Word 版说明稿。
- 为生成和结构化校验 Word 文档，在项目本地 `.venv` 中安装了 `python-docx`、`pdf2image` 和 `pillow`。
- 用户已补充完整代码、主数据集、原始裂缝图、训练日志、checkpoint、评估结果和可视化结果。
- 已只读统计新增内容：`underwater_crack_v3` 含 10,360 对 `.png` / `.png.npy`，`under-crack-images/images` 含 1,037 张 `.jpg`，`output_crackwarp` 含 13 个 checkpoint 和 gate/eval 结果。
- 已抽样检查 `underwater_crack_v3` 标签范围，样本标签 shape 为 `(2,512,512)`，范围为 `[0,1]`，与项目文档中的归一化逆映射坐标定义一致。
- 已确认当前 `.venv` 缺少训练依赖，只补充安装了用于只读标签检查的 `numpy`，尚未安装 PyTorch 或启动训练。
- 新增 `项目交接与下一步开发计划.md`，用于整文件夹拷贝到另一台电脑后继续开发，记录当前现状、已有指标、环境搭建、小闭环复现和后续优化路线。
- 新增 `run_train_slurm.py`，作为 Slurm 训练包装入口，在不改动原训练主逻辑的前提下覆盖训练数据、输出目录、epoch、worker、梯度累积和学习率等参数，并强制关闭 `restart_training` 以保护历史结果。
- 新增 `slurm_train_crackwarp.sbatch`，默认使用 `gpu` 分区、`gpo-ifv7xx` 账号、`normal` QOS、1 张 GPU、8 CPU、64GB 内存和 48 小时时间限制；训练结束后自动对 `best_epe.pth` 做 120 张样本的小规模评估。
- 更新 `.gitignore`，忽略 Slurm 标准输出和错误日志 `slurm-*.out`、`slurm-*.err`。
- 新增 `requirements.txt`，记录除 PyTorch 以外的训练、评估、可视化和文档维护依赖；PyTorch 仍需按集群 CUDA 环境单独安装 CUDA 12.x 兼容 wheel。
- 只读检查了项目下全部 6 份 Markdown 文档和 3 份 Word 文档，确认项目目标、验收口径、已有训练状态、主要短板、工作量估算和下一步 Slurm 训练需求。
- 完成一次当前项目明显问题只读体检：确认主要风险集中在现有 checkpoint 指标远未达标、`predict.py` 旧推理坐标逻辑可疑、`restart_training=True` 有清空历史输出风险、训练脚本缺少严格断点续训、文档脚本名/命令过时、`.venv` 已失效、folding 约束与评估口径仍需复核。
- 用户删除旧 `.venv` 后，已使用本机 Python 3.11 重新创建项目本地 `.venv`，安装 PyTorch `2.12.0+cu126`、torchvision `0.27.0+cu126` 和 `requirements.txt` 中的训练/评估/文档依赖；本机检测到 CUDA 可用，GPU 为 Quadro T1000。
- 新增 `smoke_train_verify.py`，用于本机最小训练验证。已在 CPU 和 GPU 上分别跑通 2 个样本、64×64、`base_ch=4`、`n_iter=0`、1 epoch 的最小训练和验证，确认数据读取、模型前向、loss、反传、验证和权重保存链路可用。
- 更新 `.gitignore`，忽略 `output_smoke_local*/` 本地 smoke 输出目录。
- 初始化本地 Git 仓库并推送到 public GitHub 仓库 `https://github.com/zh23jemu/underwater-crack-correction`；普通 Git 历史只包含代码、文档、脚本和小型说明文件，不包含大数据、历史训练输出、虚拟环境或本地 smoke 产物。
- 创建 GitHub Release `data-v1`，地址为 `https://github.com/zh23jemu/underwater-crack-correction/releases/tag/data-v1`；已上传 33 个 Release assets，总大小约 23.66GB，包括 `under_crack_images.zip`、`output_crackwarp.zip` 和 `underwater_crack_v3.tar.zst.part-001` 至 `part-031`。
- 已上传 `SHA256SUMS.txt` 到 Release，并在仓库根目录新增同名校验清单；服务器端下载后应优先执行 `sha256sum -c SHA256SUMS.txt` 验证 2 个 zip 和 31 个主数据集分卷是否完整。
- 新增并更新 `GITHUB_UPLOAD_PLAN.md`，记录 Git 内容与 Release 大文件资产划分，以及服务器侧 `gh release download`、分卷合并、完整性校验和解压命令。
- 新增 `新电脑接续工作指南.md`，用于换电脑后继续工作，集中记录 GitHub 仓库、Release 数据、服务器端数据已解压状态、新电脑 clone/pull、服务器环境重建、Slurm smoke、正式训练和后续排查路线。
- 2026-06-07 复核当前轻量仓库状态：本地已不包含大数据目录和历史输出目录，`.gitignore` 已忽略 `underwater_crack_v3/`、`under-crack-images/`、`output_crackwarp/`、`release_assets/`、Slurm 日志、checkpoint 和 smoke 产物。
- 2026-06-07 复核 Slurm 入口：`slurm_train_crackwarp.sbatch` 默认使用 `gpu` 分区、`gpo-ifv7xx` 账号、`normal` QOS、1 GPU、8 CPU、64GB 内存、48 小时，训练后会对 `best_epe.pth` 做 120 样本评估。
- 2026-06-08 根据服务器回传结果记录正式训练进展：Slurm 任务 `34827388` 已完成 50 epoch，输出到 `output_crackwarp_slurm/v2`，`.err` 为空，并生成 `eval_best_epe_120/eval_summary.json` 与 `eval_per_image.csv`。
- 2026-06-08 初步判断 v2 训练结果：`best_epe.pth` 的 120 样本 crack EPE 为 108.987938、global EPE 为 111.060646、crack Dice 为 0.283552；较旧权重略有改善但不足以支撑当前目标，需要进入诊断和结构/损失优化阶段。
- 2026-06-08 追加评估 `best_crack_epe.pth`、`best_loss.pth` 和 `best_epe.pth` 的 1000 样本结果；确认三个 best checkpoint 在 120 样本上指标一致，且 1000 样本评估显示泛化指标回落。
- 2026-06-08 追加旧模型 `output_crackwarp/best_epe.pth` 的 1000 样本复评，确认 v2 相比旧模型只是 EPE/Dice 小幅改善，edge/folding 指标退化。
- 2026-06-08 新增 `utils/compare_eval_per_image.py`，用于从新旧模型逐图评估结果中挑选诊断样本；同时更新 `utils/export_crack_roi_visuals.py`，支持通过 `--image_list` 只导出指定样本的 ROI 可视化。
- 2026-06-08 拉取服务器提交 `f6f4645`，完成对 `compare_old_vs_v2` 和共同失败 ROI 图的本地分析；确认 v2 的改进不稳定，主要问题集中在大形变、边界区域和裂缝细节保持。
- 2026-06-08 新增整图 flow 诊断脚本 `utils/export_flow_diagnostics.py`，为下一步服务器端导出 EPE/folding/位移幅度热图做准备。
- 2026-06-08 修复 `utils/export_flow_diagnostics.py` 直接运行时找不到根目录模块的问题：脚本启动时显式把项目根目录加入 `sys.path`，以兼容服务器命令 `.venv/bin/python utils/export_flow_diagnostics.py ...`。
- 2026-06-08 拉取并分析服务器生成的共同失败样本整图 flow 诊断结果；确认 v2 对全图 EPE 有局部改善，但裂缝 ROI EPE 不稳定，恢复幅度存在样本依赖的过小/过大问题。
- 2026-06-08 拉取并分析 `flow_diag_new_better` 与 `flow_diag_old_better`，确认 v2 的提升来自抑制旧模型过大位移，而退化样本多为 v2 位移幅度异常增大。
- 2026-06-08 新增可开关的裂缝 ROI 位移幅度一致性损失 `w_crack_mag`，默认关闭；Slurm 脚本支持用 `W_CRACK_MAG` 环境变量打开。
- 2026-06-08 根据服务器 v3 从零 smoke 结果修正策略：新增 `INIT_CHECKPOINT` 微调入口，避免新 loss 从随机初始化阶段主导不稳定位移场。
- 2026-06-09 记录 v3 fine-tune smoke 结果：从 v2 checkpoint 初始化、`W_CRACK_MAG=0.1`、`LR=5e-6`、2 epoch 训练链路正常，120 样本指标较 v2 有小幅改善且 folding 未爆炸。
- 2026-06-09 拉取服务器提交 `63a0971`，同步 `output_crackwarp_slurm/v3_mag_ft_smoke/eval_best_epe_1000/` 与 `compare_v2_vs_v3_ft/`，确认 v3 fine-tune 在 1000 样本上综合优于 v2，但 edge fidelity 仍小幅下降。
- 2026-06-09 根据服务器控制台输出记录 v2-v3 正反样本 flow 诊断已生成；尚需将 `flow_diag_new_better/` 与 `flow_diag_old_better/` 的 CSV 和图片同步回本地后做完整归因分析。
- 2026-06-09 拉取服务器提交 `5b9ca78`，完成 v2-v3 正反样本 flow 诊断 CSV 和代表性图片分析；确认 v3 变好组主要伴随位移幅度下降，变差组主要伴随位移幅度和 folding 小幅上升。
- 2026-06-09 拉取服务器提交 `3346ffd`，完成 `v3_mag_ft_w005_10ep` 的 1000 样本结果和逐图对比分析；确认该版本较 v2 和 `W_CRACK_MAG=0.1` smoke 更稳定，是当前主线候选。
- 2026-06-09 拉取服务器提交 `c200b54`，完成 `v3_mag_ft_w005_10ep` 退化样本和共同失败样本 flow 诊断；确认剩余失败主要集中在裂缝 ROI 的局部位移过强、folding 增加和边缘对齐不足。
- 2026-06-09 修改 `loss_crack.py`、`train_v2.py`、`run_train_slurm.py`、`slurm_train_crackwarp.sbatch` 和 `config_crack.py`，接入鲁棒位移幅度约束、过大位移惩罚和裂缝 ROI 校正图边缘一致性损失；默认关闭以保持历史结果可复现。

## Next TODO

- 在另一台电脑继续开发时，优先阅读 `项目交接与下一步开发计划.md`，按其中顺序先搭环境、做只读检查和小闭环复现。
- 在 Slurm 集群上训练前，先重新创建项目本地 `.venv`，按 CUDA 12.x 兼容策略安装 `torch torchvision` 和其余训练依赖；不要复用从 Windows/旧电脑拷贝来的 `.venv`。
- 首次 Slurm 训练建议先提交短任务或较小 epoch 复现，确认 `run_train_slurm.py` 输出目录、CUDA、数据读取和评估流程正常后，再扩大到 50-80 epoch。
- Slurm 前建议复用 `smoke_train_verify.py` 或提交 `EPOCHS=2` 的短任务做集群 smoke；确认集群 CUDA、数据路径和输出目录正常后，再提交正式训练。
- 到 Slurm 服务器后，优先 `git clone https://github.com/zh23jemu/underwater-crack-correction.git`，再使用 GitHub Release `data-v1` 下载大文件资产；主训练集需先合并 31 个 `underwater_crack_v3.tar.zst.part-*` 分卷再解压。
- 换到新电脑后，优先阅读 `新电脑接续工作指南.md`；新电脑只需 clone/pull GitHub 轻量仓库，服务器端数据已确认下载和解压，后续主要在服务器上做数据数量确认、重建 `.venv`、跑 Slurm smoke 和正式训练。
- 当前最新优先级：先在服务器执行数据数量确认和 `SHA256SUMS.txt` 校验结果复核；再重建 `.venv` 并安装 CUDA 12.x 兼容 PyTorch；随后提交 `EPOCHS=2 OUTPUT_DIR=output_crackwarp_slurm/smoke sbatch slurm_train_crackwarp.sbatch`。
- 补齐训练环境依赖：优先根据当前机器 CUDA 能力安装 PyTorch CUDA 12.x 兼容版本，再通过 `requirements.txt` 安装 `opencv-python-headless`、`scipy`、`tqdm`、`natsort`、`scikit-image` 等依赖。
- 先做不训练的运行检查：导入模型、加载 `best_epe.pth`、读取少量样本、跑一次小规模推理/评估，确认代码路径、checkpoint 结构和设备可用性。
- 修正文档与代码不一致：项目文档提到 `infer_epoch80.py`，实际文件为 `infer_epoch.py`。
- 梳理 `predict.py` 与 `infer_epoch.py` 的 remap 坐标逻辑，重点排查是否存在 `abs()`、坐标方向、归一化尺度或 `map_x/map_y` 使用不一致导致恢复量偏小。
- 在确认环境可跑后，优先做小样本诊断评估和可视化复现，再决定是否进入完整训练或结构改动。
- 如果需要继续完善对外稿，可根据实际报价策略进一步压缩为一页版、报价单版或合同附件版。
- 如果需要做 Word 视觉级 QA，需在本机安装可命令行调用的 LibreOffice `soffice`，然后重新渲染检查页面布局。
- 服务器下一步优先对 `output_crackwarp_slurm/v2/best_epe.pth` 做更完整评估：扩大到全验证集或更多样本，生成 ROI 局部放大图、误差热图、预测位移场和失败样本，避免只根据 120 张样本下结论。
- 下一轮若继续训练，不建议只沿用相同配置再跑 50 epoch；应先完成预测位移场、ROI 可视化、folding 热区和高误差样本诊断，再决定修改 loss 权重、坐标约束或局部 refinement。
- 基于 1000 样本公平对比，下一步应优先挑选旧模型优于 v2、v2 优于旧模型、两者都失败的样本各若干张，做可视化归因分析。
- 将本地新增脚本同步到服务器后，先运行 `utils/compare_eval_per_image.py` 生成三类图片清单，再分别用旧模型和 v2 模型调用 `utils/export_crack_roi_visuals.py --image_list` 导出 ROI 对比图。
- 对比旧权重与 v2 权重时统一使用当前 `utils/evaluate_metrics.py` 口径，重点比较 crack EPE、global EPE、Dice、edge fidelity 和 folding rate；旧 gate report 的 Dice 口径不应再直接混用。
- 进入优化前先检查 `infer_epoch.py`、`predict.py`、`utils/evaluate_metrics.py` 中 `cv2.remap` 坐标方向、归一化尺度、`map_x/map_y` 顺序和可视化输出是否完全一致，优先排除恢复量偏小的工程原因。
- 优先针对 `crack0030_xx` 共同失败族做专项诊断：检查合成标签的逆映射场、预测 flow 范围、边界采样、folding 热区和 ROI mask 是否对齐；这组样本可作为下一轮 loss/约束调整的回归测试集。
- 服务器拉取 `utils/export_flow_diagnostics.py` 后，先对 `joint_failure_images.txt` 导出整图诊断面板；如果 EPE 热图与 folding 热区集中在边界和大形变区域，优先调整 folding/Jacobian 约束和边界采样；如果位移幅度图整体偏小，则优先排查坐标尺度与恢复幅度约束。
- 下一步需要服务器继续对 `new_better_images.txt` 和 `old_better_images.txt` 分别导出整图 flow 诊断面板，用正反样本确认 v2 改善和退化的共同模式，再决定 loss/结构改动。
- 下一步服务器优先导出 v2-v3 的正反样本 flow 诊断图：分别使用 `output_crackwarp_slurm/v3_mag_ft_smoke/compare_v2_vs_v3_ft/new_better_images.txt` 和 `old_better_images.txt`，对比 v2 `best_epe.pth` 与 v3 `best_epe.pth` 的位移幅度、EPE 热图和 folding 热区。
- 下一步服务器先汇总 `flow_diag_new_better/flow_diagnostics.csv` 与 `flow_diag_old_better/flow_diagnostics.csv` 的均值差异，重点看 global/crack EPE、folding rate、mean/p95 displacement magnitude，再决定是否提交 10 epoch fine-tune。
- 不建议直接按当前 `W_CRACK_MAG=0.1` 跑长训 50 epoch；如果继续小规模训练，优先做更保守的 10 epoch 对照，例如 `W_CRACK_MAG=0.05` 或保留 `W_CRACK_MAG=0.1` 但增加位移幅度鲁棒项/上限项，防止高难样本的 p95 displacement magnitude 继续上升。
- 下一步不要立刻继续 50 epoch；优先对 `v3_mag_ft_w005_10ep/compare_v2_vs_w005_10ep/old_better_images.txt` 和 `joint_failure_images.txt` 导出 flow 诊断图，确认剩余 150 张退化样本是否仍由裂缝 ROI 位移过大、边界大形变或 edge fidelity 下降导致。
- 若 `w005_10ep` 的退化样本也主要是位移幅度过大，则下一轮可考虑 `W_CRACK_MAG=0.03` 的 10 epoch 对照，或在 magnitude loss 中加入 robust/clamp 策略；若主要是边缘变糊，则优先补 edge/gradient consistency，而不是继续调 `W_CRACK_MAG`。
- 下一步代码优化优先级：在 magnitude consistency 中加入鲁棒/上限策略，避免高难样本 p95 displacement magnitude 被继续推大；同时增加裂缝 ROI 边缘/梯度一致性或 edge-preserving 项，解决 EPE 改善但 edge fidelity 仍偏低的问题。
- 服务器 pull 新代码后先执行语法检查：`.venv/bin/python -m py_compile loss_crack.py train_v2.py run_train_slurm.py utils/evaluate_metrics.py`。
- 语法检查通过后建议先跑 2 epoch smoke：`INIT_CHECKPOINT=output_crackwarp_slurm/v3_mag_ft_w005_10ep/best_epe.pth W_CRACK_MAG=0.05 CRACK_MAG_ROBUST_DELTA=0.01 CRACK_MAG_OVER_WEIGHT=0.5 W_CRACK_EDGE=0.05 LR=3e-6 EPOCHS=2 OUTPUT_DIR=output_crackwarp_slurm/v4_robust_edge_smoke sbatch --partition=gpuHz --time=02:00:00 slurm_train_crackwarp.sbatch`。
- smoke 日志中需要确认 `c_mag` 和 `c_edge` 正常出现；若 120 样本评估没有明显退化，再做 1000 样本评估和 v3_w005 对比。
- 如果 v2-v3 flow 诊断确认 v3 主要是在校准位移幅度且没有引入新的边界折叠，再提交 10 epoch 小规模 fine-tune：`INIT_CHECKPOINT=output_crackwarp_slurm/v2/best_epe.pth W_CRACK_MAG=0.1 LR=5e-6 EPOCHS=10 OUTPUT_DIR=output_crackwarp_slurm/v3_mag_ft_10ep sbatch --partition=gpu --time=24:00:00 slurm_train_crackwarp.sbatch`。
- v3 后续优化不能只依赖 `w_crack_mag`；需要同步补强 edge fidelity，例如提高边缘保持项、检查 ROI 边缘 mask 对齐，或增加裂缝边缘/梯度一致性消融，否则指标会继续出现 EPE 改善但视觉边缘变差的问题。

## Open Issues

- 两份报告中的实验配置不完全一致：普通版为 80 epoch、13.14M 参数、数据总量 10370；详细版为 50 epoch、17.84M 参数、数据总量 10360。
- 普通版更强调收敛和视觉效果，详细版则指出所有 checkpoint 未达到工程门限；两份报告若要合并，需要先确认它们是否来自同一实验阶段或不同版本模型。
- 文档口径存在“普通报告较乐观、详细报告与 gate report 较严格”的差异；后续汇报时应以详细版报告和 gate report 的工程门限作为当前真实状态依据，同时保留普通版报告作为早期实验阶段记录。
- `predict.py` 中旧逻辑直接将网络输出坐标喂给 `cv2.remap`，且 `predict_cv` 使用 `np.abs(preds[:, :, 0/1])`，与 `infer_epoch.py` 中 `flow * 511` 的归一化逆映射坐标转换不一致，可能导致恢复幅度、方向和可视化判断失真。
- `loss_crack.py` 中 folding penalty 当前直接对归一化采样坐标求 `det = du_dx * dv_dy - du_dy * dv_dx` 并惩罚负值，后续需要复核它是否正确反映逆映射场相对于像素网格或恒等映射的局部可逆性；当前 gate report 的 folding rate 约 0.57，说明这条约束或其权重仍不足以控制局部折叠。
- 当前 `.venv` 已在本机重建并能完成最小 CPU/GPU smoke；但 Quadro T1000 显存和算力较弱，首次尝试 4 样本、128×128、较宽轻量模型 smoke 超过 5 分钟未完成，说明本机不适合作为正式训练环境。
- 当前 `train_v2.py` 只保存模型权重，没有保存 optimizer、scheduler、scaler 和 epoch 状态，因此还不支持严格断点续训；Slurm 中断后只能从已有权重重新初始化训练，或后续先扩展 checkpoint 结构。
- `.gitignore` 当前忽略 `*.log` 和 `*.pth`，但没有整体忽略 `underwater_crack_v3/`、`under-crack-images/`、`output_crackwarp/` 或 `.npy`；如果后续初始化 Git，需要确认哪些大目录应通过外部数据管理而不是直接入库。
- 现有 `PROJECT_WORKFLOW_GUIDE.md` / `OPTIMIZATION_GUIDE.md` 的命令使用系统 `python`，需要后续统一改为项目 `.venv` 调用方式。
- 已有 gate report 中 checkpoint 路径仍包含原机器绝对路径 `D:\nxy1\GOOD_cnn_new\GOOD_cnn\...`，后续报告或脚本应避免依赖旧路径。
- 现有模型虽然训练收敛，但 gate report 全部未达标，核心瓶颈集中在 crack EPE、Dice、edge fidelity 和 folding rate。
- 当前环境缺少 LibreOffice `soffice`，因此新增 Word 说明稿已完成结构化校验，但未完成页面 PNG 渲染级视觉 QA。
- 当前本地 Windows 工作区没有可用 `.venv`，因此 2026-06-08 新增的诊断脚本尚未在本地完成 Python 编译检查；需要同步到服务器后使用服务器 `.venv/bin/python` 做 `py_compile` 和实际运行验证。
- v3 fine-tune 在 1000 样本上虽比 v2 更稳，但改善幅度仍较小，且 edge fidelity 继续下降；当前只能证明位移幅度一致性损失有进一步实验价值，不能证明已经接近三区论文或工程验收水平。

## Architecture Decisions

- 当前将仓库定位为“水下裂缝扭曲校正算法工程与报告整理工作区”，代码、数据、训练输出和报告需要共同维护。
- 保留两份用户原始报告，不直接覆盖或改写原文件。
- `.venv/` 作为本地解析、评估和后续训练环境，已通过 `.gitignore` 排除。
- 对外工作量说明同时保留 Markdown 源稿和 Word 交付稿，便于后续快速改文字并生成客户可读版本。
- Slurm 训练采用包装入口覆盖配置，而不是复制一份训练主脚本；这样保留 `train_v2.py` 作为唯一训练逻辑源，减少后续调参时的维护分叉。
