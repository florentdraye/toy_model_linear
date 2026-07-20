#!/usr/bin/env bash
# Bilinear classification head: logits[c] = h.T @ U_c V_c.T @ h (rank=8 by default).
# Otherwise same residual transformer that hits 100% test acc with the linear head.
# The intent: see whether the *same trainable transformer* still organizes its
# residual stream linearly when the head can only read it through quadratic forms.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-bilinear-head}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 60 \
  --head-type bilinear \
  --head-rank 8 \
  --d-model 128 \
  --n-blocks 4 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
