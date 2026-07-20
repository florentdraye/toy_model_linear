#!/usr/bin/env bash
# Skipless transformer: no residual connections + ReLU^2 MLPs.
# Tracks DoM probes every K steps for three time-series:
#   rep(t)   = h(t)
#   diff(t)  = h(t) - h(prev probe step)
#   deriv(t) = h(t) - h(t - 1 training step)
# and plots them via probe_deriv.py and probe_dot.py after training.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-relu2-no-residual}"
PROBE_EVERY="${PROBE_EVERY:-100}"
PROBE_N_NODES="${PROBE_N_NODES:-20}"
BALANCE_TRAIN_LAYER="${BALANCE_TRAIN_LAYER:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-$USER}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100\
  --edges-per-node 10 \
  --lr 1e-3 \
  --weight-decay 1.0 \
  --batch-size 4000 \
  --n-epochs 40 \
  --mlp-activation relu2 \
  --n-blocks 4 \
  --balance-train-layer "$BALANCE_TRAIN_LAYER" \
  --dom-probe-every-steps "$PROBE_EVERY" \
  --dom-probe-n-nodes "$PROBE_N_NODES" \
  --out-dir "$OUT_DIR" \
  # --no-use-residual \
  # --suo-init \
  # --optimizer klshampoo \

  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

LAYERS="${LAYERS:-3 4}"
python -u probe_deriv.py --out-dir "$OUT_DIR" --layers $LAYERS 2>&1 | tee -a "$OUT_DIR/log.txt"
python -u probe_dot.py   --out-dir "$OUT_DIR" --layers $LAYERS 2>&1 | tee -a "$OUT_DIR/log.txt"

echo "$OUT_DIR"
