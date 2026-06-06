# GitHub 上传规划

本文档记录本项目推送到 GitHub 时的文件划分，避免把 20GB 级训练数据或 checkpoint 误提交进 Git 历史。

## 1. 进入 Git 仓库的内容

以下内容体量较小，适合进入普通 Git 仓库：

- Python 源码：`*.py`
- 模型与工具源码：`models/`、`utils/`
- 项目说明与交接文档：`*.md`
- Word 报告与工作量说明：`*.docx`
- 环境与提交脚本：`requirements.txt`、`slurm_train_crackwarp.sbatch`
- 本地最小训练验证脚本：`smoke_train_verify.py`
- 项目维护记录：`AGENTS.md`

## 2. 不进入 Git 的大文件/本地运行产物

以下内容应通过 GitHub Release assets 或其它对象存储分发，不进入 Git：

| 路径 | 估算体量 | 用途 | 建议处理 |
|---|---:|---|---|
| `underwater_crack_v3/` | 约 24.41GB | 主合成训练集 | 分卷压缩后上传 Release |
| `output_crackwarp/` | 约 0.92GB | 历史 checkpoint、评估和可视化结果 | 压缩后上传 Release |
| `under-crack-images/` | 约 0.05GB | 原始裂缝图 | 压缩后上传 Release |
| `.venv/` | 约 4.57GB | 本机虚拟环境 | 不上传，按 `requirements.txt` 重建 |
| `output_smoke_local*/` | 小型本机验证产物 | smoke test 输出 | 不上传，必要时重新生成 |

## 3. GitHub Release 资产建议

GitHub Release 单个资产需要控制在 2GB 以下，因此最大的数据集应分卷。建议资产命名如下：

- `underwater_crack_v3.tar.zst.part-001`
- `underwater_crack_v3.tar.zst.part-002`
- ...
- `under_crack_images.tar.zst`
- `output_crackwarp.tar.zst`

如果本机没有 `zstd` 分卷工具，也可以使用 `tar` 生成 `.tar` 后再分卷，或在服务器端使用 `gh release download` 下载所有分卷后合并。

## 4. 服务器侧获取方式

轻量代码：

```bash
git clone <仓库地址>
cd underwater-crack-correction
```

Release 资产：

```bash
gh release download data-v1 --repo <owner>/<repo> --dir release_assets
```

下载后按实际资产格式解压到项目根目录，再运行 smoke 或 Slurm 训练。

