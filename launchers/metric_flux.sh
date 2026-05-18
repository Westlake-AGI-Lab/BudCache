#!/bin/bash
set -euo pipefail

export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false

EVAL_RUN_DIR=${EVAL_RUN_DIR:-}
PROMPTS_FILE=${PROMPTS_FILE:-data/eval/drawbench.txt}

python -m src.metric.image_quality_metric "$EVAL_RUN_DIR" "$PROMPTS_FILE"
