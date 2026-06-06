#!/bin/bash
#SBATCH --job-name=crackwarp-train
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Slurm 训练提交脚本。
# 默认策略：
# - GPU 分区使用 gpu，避免默认走 aws 产生额外费用。
# - 账号/QOS 使用项目约定的 gpo-ifv7xx / normal。
# - Python 必须使用项目本地 .venv，不依赖激活虚拟环境。
# - 输出目录默认按 Slurm Job ID 分开，避免覆盖历史 output_crackwarp。
#
# 可在提交时覆盖分区，例如：
#   sbatch --partition=gpuHz slurm_train_crackwarp.sbatch
#
# 可在提交时覆盖训练参数，例如：
#   EPOCHS=80 WORKERS=12 OUTPUT_DIR=output_crackwarp_slurm/v2 sbatch slurm_train_crackwarp.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

PYTHON_BIN=".venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[错误] 未找到 ${PYTHON_BIN}。请先在集群项目目录创建 .venv 并安装依赖。"
  echo "建议 PyTorch GPU wheel 使用 CUDA 12.x 兼容版本，例如 cu126。"
  exit 1
fi

# 让 DataLoader、OpenMP 和 Slurm CPU 分配保持一致，减少无意义的线程争用。
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

TRAINROOT="${TRAINROOT:-underwater_crack_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-output_crackwarp_slurm/${SLURM_JOB_ID:-manual}}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
LR="${LR:-2e-5}"

echo "========== CrackWarp Slurm 训练 =========="
echo "提交目录: ${PWD}"
echo "训练数据: ${TRAINROOT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Workers: ${WORKERS}"
echo "Accum steps: ${ACCUM_STEPS}"
echo "LR: ${LR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
echo "=========================================="

"${PYTHON_BIN}" -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

"${PYTHON_BIN}" run_train_slurm.py \
  --trainroot "${TRAINROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --accum-steps "${ACCUM_STEPS}" \
  --lr "${LR}" \
  --gpu-id 0

# 训练结束后做一轮小规模门限前诊断，方便快速判断这次训练有没有方向性改善。
# 这里默认评估 120 张样本，完整评估可后续单独提交更长任务。
if [ -f "${OUTPUT_DIR}/best_epe.pth" ]; then
  "${PYTHON_BIN}" utils/evaluate_metrics.py \
    --model "${OUTPUT_DIR}/best_epe.pth" \
    --img_dir "${TRAINROOT}" \
    --out_dir "${OUTPUT_DIR}/eval_best_epe_120" \
    --num 120 \
    --batch_size 1 \
    --gpu 0
fi
