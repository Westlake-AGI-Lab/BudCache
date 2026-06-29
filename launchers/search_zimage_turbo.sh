#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
# === launch ===
# Requires MODEL_ROOT to be exported (model is loaded from ${MODEL_ROOT}/${model_name}).
export RUN_ID=$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}${SUFFIX:+_$SUFFIX}


CONFIG=${CONFIG:-configs/search_zimage_turbo.yaml}

CUDA_VISIBLE_DEVICES=1 python -m scripts.stage1_search_zimage "${CONFIG}"
