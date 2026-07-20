#!/usr/bin/env bash
# Replace Q and K projections with bilinear MLPs while keeping V linear and the
# rest of the model standard. The attention logits are therefore
#   logits_{ij} = (Q(x_i)) . (K(x_j))
# where Q and K are quadratic in x_i and x_j respectively, so the logits are
# bilinear in (q_features(x_i), k_features(x_j)) which makes the *attention pattern*
# a 4th-order function of the inputs.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-qk-bilinear}"
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
  --attn-qk-mlp \
  --mlp-type bilinear \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
