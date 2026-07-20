#!/usr/bin/env bash
# 2000-step relu2 training, DoM + MLP probes every 100 steps.
# ~80s wall clock target (MLP is the heaviest probe).
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-2k-relu2-mlp}"
mkdir -p "$OUT_DIR"

/home/fdraye/.pyenv/versions/3.11.0/bin/python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 21 \
  --mlp-activation relu2 \
  --dom-probe-every-steps 100 \
  --dom-probe-n-nodes 20 \
  --dom-mlp-hidden 64 \
  --dom-mlp-iters 300 \
  --dom-mlp-lr 3e-2 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

echo "$OUT_DIR"
