#!/usr/bin/env bash
set -euo pipefail

# Roboflow 两个外部候选评估集的补充横评脚本。
#
# 目标：
# - 针对 v108 数据消融中已经生成好的两个 Roboflow 合成评估集；
# - 补齐 4 个外部 optical-flow / dense-matching 参考模型 + v107 主模型；
# - 输出统一 eval_summary.json、eval_per_image.csv、汇总 CSV/Markdown；
# - 只写入新的输出目录，不覆盖历史训练、checkpoint 或主数据集评估结果。
#
# 说明：
# - 外部模型采用 oracle-pair 设置：使用 GT 校正图与输入图进行 dense matching；
# - GMA 如果缺少官方仓库或 checkpoint，则继续采用既定口径 RAFT-small fallback；
# - v107 为单图像校正模型，和外部 oracle-pair baseline 输入条件不同，报告中需注明。

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-output_crackwarp_slurm/roboflow_external_v107_eval}"
GPU_ID="${GPU_ID:-0}"
NUM="${NUM:--1}"
SIZE="${SIZE:-512}"
ALLOW_GMA_FALLBACK="${ALLOW_GMA_FALLBACK:-1}"

UNDERWATER_DIR="${UNDERWATER_DIR:-data_candidates/v108_data_ablation/generated/roboflow_underwater_crack_sp2}"
CONCRETE_DIR="${CONCRETE_DIR:-data_candidates/v108_data_ablation/generated/roboflow_concrete_crack_small_sp2}"

UNIMATCH_REPO="${UNIMATCH_REPO:-external_methods/unimatch}"
UNIMATCH_CKPT="${UNIMATCH_CKPT:-external_methods/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata.pth}"
SEARAFT_REPO="${SEARAFT_REPO:-external_methods/SEA-RAFT}"
SEARAFT_CFG="${SEARAFT_CFG:-external_methods/SEA-RAFT/config/eval/spring-M.json}"
SEARAFT_CKPT="${SEARAFT_CKPT:-}"
SEARAFT_URL="${SEARAFT_URL:-MemorySlices/Tartan-C-T-TSKH-spring540x960-M}"
GMA_REPO="${GMA_REPO:-external_methods/GMA}"
GMA_CKPT="${GMA_CKPT:-external_methods/GMA/checkpoints/gma-sintel.pth}"

V107_CKPT="${V107_CKPT:-output_crackwarp_slurm/v107_latest_method_long100_from_v106_r2/best_epe.pth}"
V107_MODEL="${V107_MODEL:-global_context_flow_unet_image_detail_head_strong}"

STATUS_LOG="${OUT_ROOT}/run_status.log"

mkdir -p "${OUT_ROOT}"
: > "${STATUS_LOG}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_LOG}"
}

run_or_mark() {
  local name="$1"
  shift
  log "START ${name}"
  if "$@" 2>&1 | tee -a "${STATUS_LOG}"; then
    log "DONE ${name}"
  else
    log "FAILED ${name}"
    return 0
  fi
}

count_pairs() {
  local img_dir="$1"
  "${PYTHON_BIN}" - "${img_dir}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
count = 0
for name in sorted(os.listdir(root)):
    if name.lower().endswith((".png", ".jpg", ".jpeg")) and (root / f"{name}.npy").exists():
        count += 1
print(count)
PY
}

eval_external() {
  local dataset_key="$1"
  local img_dir="$2"
  local method_key="$3"
  local method_name="$4"
  local pred_dir="$5"
  local eval_dir="$6"

  if [ ! -d "${pred_dir}" ]; then
    log "SKIP eval ${dataset_key}/${method_name}: 缺少预测目录 ${pred_dir}"
    return 0
  fi
  run_or_mark "eval ${dataset_key}/${method_name}" "${PYTHON_BIN}" utils/evaluate_flow_predictions.py \
    --method_name "${method_key}" \
    --pred_dir "${pred_dir}" \
    --img_dir "${img_dir}" \
    --out_dir "${eval_dir}" \
    --num "${NUM}" \
    --size "${SIZE}" \
    --batch_size 4 \
    --gpu "${GPU_ID}"
}

run_dataset() {
  local dataset_key="$1"
  local img_dir="$2"
  local dataset_out="${OUT_ROOT}/${dataset_key}"
  local pair_count

  if [ ! -d "${img_dir}" ]; then
    log "SKIP ${dataset_key}: 数据目录不存在 ${img_dir}"
    return 0
  fi

  pair_count="$(count_pairs "${img_dir}")"
  log "DATASET ${dataset_key}: IMG_DIR=${img_dir}, pairs=${pair_count}, NUM=${NUM}"
  if [ "${pair_count}" = "0" ]; then
    log "SKIP ${dataset_key}: 没有找到 .png/.npy 配对"
    return 0
  fi

  mkdir -p "${dataset_out}"

  local raft_pred="${dataset_out}/raft/pred_grid"
  local gma_pred="${dataset_out}/gma/pred_grid"
  local unimatch_pred="${dataset_out}/unimatch/pred_grid"
  local searaft_pred="${dataset_out}/searaft/pred_grid"

  run_or_mark "export ${dataset_key}/RAFT-large" "${PYTHON_BIN}" utils/export_torchvision_raft_predictions.py \
    --img_dir "${img_dir}" \
    --out_dir "${raft_pred}" \
    --variant large \
    --num "${NUM}" \
    --size "${SIZE}" \
    --gpu "${GPU_ID}" \
    --overwrite
  eval_external "${dataset_key}" "${img_dir}" "raft_oracle_pair" "RAFT" "${raft_pred}" "${dataset_out}/raft/eval"

  if [ -f "${GMA_CKPT}" ] && [ -d "${GMA_REPO}" ]; then
    run_or_mark "export ${dataset_key}/GMA" "${PYTHON_BIN}" utils/export_gma_predictions.py \
      --img_dir "${img_dir}" \
      --out_dir "${gma_pred}" \
      --gma_repo "${GMA_REPO}" \
      --checkpoint "${GMA_CKPT}" \
      --num "${NUM}" \
      --size "${SIZE}" \
      --gpu "${GPU_ID}" \
      --overwrite
    eval_external "${dataset_key}" "${img_dir}" "gma_oracle_pair" "GMA" "${gma_pred}" "${dataset_out}/gma/eval"
  elif [ "${ALLOW_GMA_FALLBACK}" = "1" ]; then
    log "GMA checkpoint/repo missing，${dataset_key} 启用 RAFT-small fallback。"
    run_or_mark "export ${dataset_key}/RAFT-small fallback" "${PYTHON_BIN}" utils/export_torchvision_raft_predictions.py \
      --img_dir "${img_dir}" \
      --out_dir "${gma_pred}" \
      --variant small \
      --num "${NUM}" \
      --size "${SIZE}" \
      --gpu "${GPU_ID}" \
      --overwrite
    eval_external "${dataset_key}" "${img_dir}" "raft_small_gma_slot_fallback" "GMA/RAFT-small fallback" "${gma_pred}" "${dataset_out}/gma/eval"
  else
    log "SKIP ${dataset_key}/GMA: 缺少 ${GMA_REPO} 或 ${GMA_CKPT}"
  fi

  if [ -d "${UNIMATCH_REPO}" ] && [ -f "${UNIMATCH_CKPT}" ]; then
    run_or_mark "export ${dataset_key}/UniMatch" "${PYTHON_BIN}" utils/export_unimatch_predictions.py \
      --img_dir "${img_dir}" \
      --out_dir "${unimatch_pred}" \
      --unimatch_repo "${UNIMATCH_REPO}" \
      --checkpoint "${UNIMATCH_CKPT}" \
      --num "${NUM}" \
      --size "${SIZE}" \
      --gpu "${GPU_ID}" \
      --overwrite
    eval_external "${dataset_key}" "${img_dir}" "unimatch_oracle_pair" "UniMatch" "${unimatch_pred}" "${dataset_out}/unimatch/eval"
  else
    log "SKIP ${dataset_key}/UniMatch: 缺少 ${UNIMATCH_REPO} 或 ${UNIMATCH_CKPT}"
  fi

  if [ -d "${SEARAFT_REPO}" ] && [ -f "${SEARAFT_CFG}" ]; then
    run_or_mark "export ${dataset_key}/SEA-RAFT" "${PYTHON_BIN}" utils/export_searaft_predictions.py \
      --img_dir "${img_dir}" \
      --out_dir "${searaft_pred}" \
      --searaft_repo "${SEARAFT_REPO}" \
      --cfg "${SEARAFT_CFG}" \
      --checkpoint "${SEARAFT_CKPT}" \
      --url "${SEARAFT_URL}" \
      --num "${NUM}" \
      --size "${SIZE}" \
      --gpu "${GPU_ID}" \
      --overwrite
    eval_external "${dataset_key}" "${img_dir}" "searaft_oracle_pair" "SEA-RAFT" "${searaft_pred}" "${dataset_out}/searaft/eval"
  else
    log "SKIP ${dataset_key}/SEA-RAFT: 缺少 ${SEARAFT_REPO} 或 ${SEARAFT_CFG}"
  fi

  if [ -f "${V107_CKPT}" ]; then
    run_or_mark "eval ${dataset_key}/v107" "${PYTHON_BIN}" utils/evaluate_metrics.py \
      --model "${V107_CKPT}" \
      --model_name "${V107_MODEL}" \
      --img_dir "${img_dir}" \
      --out_dir "${dataset_out}/v107/eval" \
      --num "${NUM}" \
      --batch_size 4 \
      --gpu "${GPU_ID}"
  else
    log "SKIP ${dataset_key}/v107: 缺少 checkpoint ${V107_CKPT}"
  fi
}

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[ERROR] Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi

log "Roboflow 外部四模型 + v107 补评估开始"
run_dataset "roboflow_underwater_crack" "${UNDERWATER_DIR}"
run_dataset "roboflow_concrete_blue_crack" "${CONCRETE_DIR}"

run_or_mark "summarize roboflow external v107 eval" "${PYTHON_BIN}" scripts/summarize_roboflow_external_v107_eval.py \
  --out-root "${OUT_ROOT}"

{
  echo "# Roboflow 外部四模型 + v107 补评估"
  echo
  echo "- status: completed_or_partially_completed"
  echo "- out_root: ${OUT_ROOT}"
  echo "- summary_csv: ${OUT_ROOT}/roboflow_external_v107_metrics.csv"
  echo "- summary_md: ${OUT_ROOT}/roboflow_external_v107_metrics.md"
  echo "- log: ${STATUS_LOG}"
} > "${OUT_ROOT}/ROBOFLOW_EXTERNAL_V107_EVAL_DONE.md"

log "Roboflow 外部四模型 + v107 补评估结束"
