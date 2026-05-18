#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export RUN_ID=$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}${SUFFIX:+_$SUFFIX}

CONFIG=${CONFIG:-configs/eval_wan_budcache.yaml}
STAGE1_CKPT=${STAGE1_CKPT:-}
STAGE2_CKPT=${STAGE2_CKPT:-}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

ARGS=("${CONFIG}" --stage1-ckpt "${STAGE1_CKPT}")
if [[ -n "${STAGE2_CKPT}" ]]; then
  ARGS+=(--stage2-ckpt "${STAGE2_CKPT}")
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --module scripts.eval_wan_budcache \
  "${ARGS[@]}"
