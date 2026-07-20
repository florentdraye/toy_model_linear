#!/usr/bin/env bash
# Skipless transformer: no residuals, ReLU^2 MLPs, AND the attention V projection
# is replaced by a 2-layer MLP (same activation = ReLU^2, hidden dim = d_model).
# Other knobs match the working run_relu2_no_residual.sh: SUO/per-head ortho
# init, KL-Shampoo, weight_decay = 1.0 (which is what made the skipless run
# actually converge to 100% test acc).
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-relu2-no-residual-mlp-v}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 60 \
  --mlp-activation relu2 \
  --no-use-residual \
  --suo-init \
  --optimizer klshampoo \
  --attn-v-mlp \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
