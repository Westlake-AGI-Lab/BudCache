#!/bin/bash
set -euo pipefail

export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
# === launch ===
export RUN_ID=$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}${SUFFIX:+_$SUFFIX}

CONFIG=${CONFIG:-configs/train_flux.yaml}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

accelerate launch \
  --num_machines 1 \
  --num_processes "${GPUS_PER_NODE}" \
  --num_cpu_threads_per_process 16 \
  --main_process_port 23457 \
  --module scripts.stage2_train_schedule_flux \
  --config "${CONFIG}"
