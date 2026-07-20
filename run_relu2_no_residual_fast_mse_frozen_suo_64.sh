#!/usr/bin/env bash
# 64-d residual stream with a frozen semi-orthogonal 100-class unembedding.
# Otherwise matches run_relu2_no_residual_fast_mse_identity.sh.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-relu2-no-residual-fast-mse-frozen-suo-d64}"
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
  --d-model 64 \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --batch-size 4000 \
  --n-epochs 120 \
  --mlp-activation relu2 \
  --n-blocks 4 \
  --balance-train-layer "$BALANCE_TRAIN_LAYER" \
  --frozen-suo-unembed \
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
