#!/bin/bash
set -euo pipefail

export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
# === launch ===
export RUN_ID=$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}${SUFFIX:+_$SUFFIX}

CONFIG=${CONFIG:-configs/eval_flux_budcache.yaml}
STAGE1_CKPT=${STAGE1_CKPT:-stage1-ckpt/flux_tab1_28nfe10.pt}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --module scripts.eval_flux_budcache \
  --config "${CONFIG}" \
  --stage1-ckpt "${STAGE1_CKPT}"
