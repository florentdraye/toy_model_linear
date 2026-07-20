#!/usr/bin/env bash
# Train with soft semi-orthogonality penalty on every nn.Linear weight.
# Pair with `python probe.py --ckpt <out>/ckpt.pt` to see whether latents stay linear.
# Override on the CLI, e.g. ./run_ortho.sh --ortho-lambda 3e-3 --n-epochs 60
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-ortho}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 200 \
  --ortho-lambda 2e-2 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
