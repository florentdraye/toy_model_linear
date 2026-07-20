#!/usr/bin/env bash
# Bilinear head + d_model bottlenecked below latent cardinality (32 < 100).
# Combined effect:
#  - bilinear head removes the "answer = direction" gradient pressure
#  - d_model=32 < #reachable_per_layer makes single-direction-per-class impossible
# Goal: 100% test acc but linear probes should drop materially on every layer.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-bilinear-bottleneck}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 120 \
  --d-model 32 \
  --n-heads 4 \
  --d-ff 128 \
  --n-blocks 4 \
  --head-type bilinear \
  --head-rank 8 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
