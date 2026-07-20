#!/usr/bin/env bash
# Adversarial-probe training: linear probes at every (depth, position) for every
# INTERMEDIATE graph layer are jointly trained, with gradient reversal pushing
# the model to make h LESS linearly decodable. The model still uses the standard
# residual transformer + linear head, so it can still hit 100% test acc - the
# question is whether intermediate AUCs come down.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-adv-probe}"
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
  --adv-probe-lambda 0.3 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
