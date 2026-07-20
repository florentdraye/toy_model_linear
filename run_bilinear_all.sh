#!/usr/bin/env bash
# Every MLP in the model is replaced by a bilinear MLP (Linear x Linear -> Linear,
# elementwise product as the only nonlinearity). Combined with the bilinear head.
# d_model stays 128; this is the "no smooth nonlinearity anywhere; everything is
# polynomial in the input" version.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-bilinear-all}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 60 \
  --d-model 128 \
  --d-ff 512 \
  --n-blocks 4 \
  --mlp-type bilinear \
  --head-type bilinear \
  --head-rank 8 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
