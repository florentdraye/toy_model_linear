#!/usr/bin/env bash
# Variant of run_relu2_no_residual_fast.sh with:
#   --identity-unembed   : ln_f and head replaced with nn.Identity (logits = h[4,5])
#   --loss-type mse      : MSE against one-hot targets, no softmax anywhere
#
# Point of this run: with no softmax during training, the model is pushed to
# make h[4,5] ~ one_hot(y) directly, so the linear DoM probe on the readout
# tensor should track the model's own argmax accuracy (~1:1 on layer 6).
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-relu2-no-residual-fast-mse-identity}"
PROBE_EVERY="${PROBE_EVERY:-500}"
PROBE_N_NODES="${PROBE_N_NODES:-10}"
PROBE_MAX_TRAIN="${PROBE_MAX_TRAIN:-10000}"
PROBE_MAX_TEST="${PROBE_MAX_TEST:-1500}"
BALANCE_TRAIN_LAYER="${BALANCE_TRAIN_LAYER:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --batch-size 4000 \
  --n-epochs 40 \
  --mlp-activation relu2 \
  --n-blocks 4 \
  --balance-train-layer "$BALANCE_TRAIN_LAYER" \
  --identity-unembed \
  --loss-type mse \
  --dom-probe-every-steps "$PROBE_EVERY" \
  --dom-probe-n-nodes "$PROBE_N_NODES" \
  --dom-probe-max-train "$PROBE_MAX_TRAIN" \
  --dom-probe-max-test "$PROBE_MAX_TEST" \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

LAYERS="${LAYERS:-3 4}"
python -u probe_deriv.py --out-dir "$OUT_DIR" --layers $LAYERS 2>&1 | tee -a "$OUT_DIR/log.txt"
python -u probe_dot.py   --out-dir "$OUT_DIR" --layers $LAYERS 2>&1 | tee -a "$OUT_DIR/log.txt"

echo "$OUT_DIR"
