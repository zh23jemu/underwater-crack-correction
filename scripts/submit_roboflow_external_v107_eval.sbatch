#!/usr/bin/env bash
#SBATCH --job-name=roboflow-ext-v107
#SBATCH --partition=gpu
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

# Roboflow 外部四模型 + v107 补评估 Slurm 入口。
# 可在提交时用环境变量覆盖：
#   NUM=-1 OUT_ROOT=... sbatch scripts/submit_roboflow_external_v107_eval.sbatch

cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs/logs

module load slurm >/dev/null 2>&1 || true

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export NUM="${NUM:--1}"
export OUT_ROOT="${OUT_ROOT:-output_crackwarp_slurm/roboflow_external_v107_eval}"
export GPU_ID="${GPU_ID:-0}"

bash scripts/run_roboflow_external_v107_eval.sh
