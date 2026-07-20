#!/usr/bin/env bash
# Same as run_relu2_no_residual.sh but with differentiable DoM erasure of K
# sampled classes at graph layer ERASE_LAYER: on each training batch, the DoM
# direction for each chosen class is projected out of the residual stream at
# every (depth, position). Because DoM is computed from the batch's own hidden
# states, gradients flow through the projection -- the model must actually
# make those latents linearly invisible, not just game a fixed direction.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-relu2-no-residual-erase}"
PROBE_EVERY="${PROBE_EVERY:-100}"
PROBE_N_NODES="${PROBE_N_NODES:-20}"
BALANCE_TRAIN_LAYER="${BALANCE_TRAIN_LAYER:-4}"
ERASE_LAYER="${ERASE_LAYER:-4}"
ERASE_N_FEATURES="${ERASE_N_FEATURES:-8}"
ERASE_SEED="${ERASE_SEED:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100\
  --edges-per-node 10 \
  --lr 2e-3 \
  --weight-decay 1.0 \
  --batch-size 4000 \
  --n-epochs 300 \
  --mlp-activation relu2 \
  --n-blocks 4 \
  --balance-train-layer "$BALANCE_TRAIN_LAYER" \
  --dom-probe-every-steps "$PROBE_EVERY" \
  --dom-probe-n-nodes "$PROBE_N_NODES" \
  --erase-layer "$ERASE_LAYER" \
  --erase-n-features "$ERASE_N_FEATURES" \
  --erase-seed "$ERASE_SEED" \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

LAYERS="${LAYERS:-3 4}"
python -u probe_deriv.py --out-dir "$OUT_DIR" --layers $LAYERS 2>&1 | tee -a "$OUT_DIR/log.txt"
python -u probe_dot.py   --out-dir "$OUT_DIR" --layers $LAYERS 2>&1 | tee -a "$OUT_DIR/log.txt"

echo "$OUT_DIR"
