#!/usr/bin/env bash
# Same as run_fast.sh but transformer blocks are attention-only (no MLP).
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-no-mlp}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 400 \
  --no-use-mlp \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"