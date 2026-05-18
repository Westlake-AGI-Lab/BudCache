#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
# === launch ===
export RUN_ID=$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}${SUFFIX:+_$SUFFIX}

CONFIG=${CONFIG:-configs/search_flux.yaml}

python -m scripts.stage1_search_flux "${CONFIG}"
