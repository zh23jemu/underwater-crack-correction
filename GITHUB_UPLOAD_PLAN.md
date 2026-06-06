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
git clone https://github.com/zh23jemu/underwater-crack-correction.git
cd underwater-crack-correction
```

Release 资产：

```bash
gh release download data-v1 --repo zh23jemu/underwater-crack-correction --dir release_assets
```

下载后按实际资产格式解压到项目根目录，再运行 smoke 或 Slurm 训练。

当前 Release 地址：

- https://github.com/zh23jemu/underwater-crack-correction/releases/tag/data-v1

当前 Release 资产：

- `under_crack_images.zip`：原始裂缝图。
- `output_crackwarp.zip`：已有训练输出、checkpoint、评估和可视化结果。
- `underwater_crack_v3.tar.zst.part-001` 到 `underwater_crack_v3.tar.zst.part-031`：主合成训练集分卷。

在 Linux/Slurm 服务器上合并并解压主训练集：

```bash
mkdir -p release_assets
gh release download data-v1 --repo zh23jemu/underwater-crack-correction --dir release_assets

cat release_assets/underwater_crack_v3.tar.zst.part-* > release_assets/underwater_crack_v3.tar.zst
tar -xf release_assets/underwater_crack_v3.tar.zst
unzip release_assets/under_crack_images.zip
unzip release_assets/output_crackwarp.zip
```

如果服务器没有 `gh`，也可以在 Release 页面手动下载所有资产，或使用 GitHub API 下载。

## 5. 下载完整性校验

如果解压时报错，先不要急着重下全部文件，应先校验下载完整性：

```bash
cd underwater-crack-correction

# 如果 SHA256SUMS.txt 来自 Git 仓库根目录，先复制到 release_assets。
cp SHA256SUMS.txt release_assets/

cd release_assets
sha256sum -c SHA256SUMS.txt
```

期望所有条目均显示 `OK`。如果只有某一个 `part-xxx` 失败，只需要重新下载那个分卷。

也可以快速确认分卷数量：

```bash
ls release_assets/underwater_crack_v3.tar.zst.part-* | wc -l
```

期望输出为 `31`。

合并后可先测试压缩包是否能被识别：

```bash
cat release_assets/underwater_crack_v3.tar.zst.part-* > release_assets/underwater_crack_v3.tar.zst
tar -tf release_assets/underwater_crack_v3.tar.zst >/dev/null
```

如果 `sha256sum -c` 全部通过，但 `tar -tf` 报不认识 zstd 或类似错误，通常是服务器 `tar` 不支持 zstd。可改用：

```bash
zstd -d release_assets/underwater_crack_v3.tar.zst -o release_assets/underwater_crack_v3.tar
tar -xf release_assets/underwater_crack_v3.tar
```

如果服务器没有 `zstd`，可在集群环境中安装或加载对应模块。
