#!/bin/bash
set -euo pipefail

EVAL_RUN_DIR=${EVAL_RUN_DIR:-}

python -m src.metric.video_recon_metric "${EVAL_RUN_DIR}"
