#!/usr/bin/env bash
# Same as run_no_mlp.sh (attention-only transformer blocks), plus DoM + LogReg
# probe tracking so you can generate the decodability plots.
#
# After training:
#   python plot_dom_layer.py --run <OUT_DIR> --layer 4
#   python plot_dom_layer.py --run <OUT_DIR> --layer 6
#   python probe_final.py    --run <OUT_DIR>          # post-hoc DoM vs LogReg grids
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-no-mlp-dom}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 1.0 \
  --batch-size 8192 \
  --n-epochs 60 \
  --no-use-mlp \
  --dom-probe-every-steps 50 \
  --dom-probe-n-nodes 20 \
  --dom-probe-seed 0 \
  --dom-logreg-iters 200 \
  --dom-logreg-lr 3e-2 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

echo
echo "Suggested next steps:"
echo "  python plot_dom_layer.py --run $OUT_DIR --layer 4"
echo "  python plot_dom_layer.py --run $OUT_DIR --layer 6"
echo "  python probe_final.py    --run $OUT_DIR"
