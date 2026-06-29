set -e
# === Env ===
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
# === launch ===
export MODEL_ROOT= # your models_path
export RUN_ID=$(date +%Y%m%d_%H%M%S)${SLURM_JOB_ID:+_job$SLURM_JOB_ID}${SUFFIX:+_$SUFFIX}


CONFIG=${CONFIG:-configs/eval_zimage_turbo.yaml}
STAGE1_CKPT=${STAGE1_CKPT:-stage1-ckpt/zimage_turbo_8nfe4.pt}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --module scripts.eval_zimage_budcache \
  --config "${CONFIG}" \
  --stage1-ckpt "${STAGE1_CKPT}"
