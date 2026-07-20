#!/usr/bin/env bash
# Same as run_fast.sh but with a 4-layer frozen, random-init MLP (relu^2)
# inserted between the transformer's ln_f and the (still trainable) classification head.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-frozen-mlp}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 120 \
  --frozen-mlp-layers 4 \
  --frozen-mlp-activation relu2 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
