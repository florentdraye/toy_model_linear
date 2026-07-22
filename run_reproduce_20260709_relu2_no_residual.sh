#!/usr/bin/env bash
# Exact effective configuration recovered from
# runs/20260709-104912-relu2-no-residual/ckpt.pt.
# Note: residual connections were enabled in that run despite its directory name.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-repro-20260709-relu2-no-residual}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR" "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --d-model 128 \
  --n-heads 4 \
  --d-ff 512 \
  --n-blocks 4 \
  --mlp-activation relu2 \
  --use-residual \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 16000 \
  --n-epochs 40 \
  --balance-train-layer 4 \
  --dom-probe-every-steps 100 \
  --dom-probe-n-nodes 20 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

python -u probe_deriv.py --out-dir "$OUT_DIR" --layers 3 4 2>&1 | tee -a "$OUT_DIR/log.txt"
python -u probe_dot.py --out-dir "$OUT_DIR" --layers 3 4 2>&1 | tee -a "$OUT_DIR/log.txt"

echo "$OUT_DIR"
