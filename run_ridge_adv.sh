#!/usr/bin/env bash
# Closed-form ridge adversary: at every (depth, position, intermediate graph
# layer) the per-batch optimal ridge fit quality is added to the loss. Forces
# latents to be non-linearly-decodable while the model still has to solve the task.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-ridge-adv}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --d-model 128 \
  --d-ff 512 \
  --n-blocks 4 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 200 \
  --ridge-adv-lambda 1.0 \
  --ridge-adv-nu 1e-3 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
