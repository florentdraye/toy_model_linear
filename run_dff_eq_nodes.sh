#!/usr/bin/env bash
# Fast preset but with MLP hidden dim = number of nodes per graph layer (100).
# So each block's MLP is Linear(d_model=128 -> d_ff=100) -> act -> Linear(100 -> 128).
# Override d_model/d_ff/etc. on the CLI as usual.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-dff100}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 120 \
  --d-ff 100 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
