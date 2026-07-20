#!/usr/bin/env bash
# Same as run_relu2_no_residual.sh, plus DoM-probe tracking.
# Saves runs/<name>/dom_probe.pt containing the full (T, D, L, ...) probe cubes
# and ckpt.pt with model state + graph + split, which is everything needed for:
#   python plot_dom_layer.py --run <OUT_DIR> --layer <N>   # decodability over training
#   python probe_layer4.py   --run <OUT_DIR> --target-layer <N>   # final-model per-cell
# where <N> is the graph layer index (0..n_layers-1).
#
# NOTE: this script matches run_relu2_no_residual.sh as-written -- residual is
# still ON, optimizer is still AdamW, no SUO init. To get the actual skipless
# stack described in that script's comments, also add:
#   --no-use-residual --suo-init --optimizer klshampoo
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-relu2-no-residual-dom}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 0.0 \
  --batch-size 8000 \
  --n-epochs 240 \
  --mlp-activation relu2 \
  --dom-probe-every-steps 50 \
  --dom-probe-n-nodes 20 \
  --dom-probe-seed 0 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

echo
echo "Training done. Artifacts in $OUT_DIR:"
echo "  ckpt.pt        model state + configs + split (for probe_layer4.py)"
echo "  dom_probe.pt   probe cubes over training (for plot_dom_layer.py)"
echo "  graph.pt       graph structure"
echo "  log.txt        training log"
echo
echo "Suggested next steps:"
echo "  python plot_dom_layer.py --run $OUT_DIR --layer 4"
echo "  python plot_dom_layer.py --run $OUT_DIR --layer 6"
echo "  python probe_layer4.py   --run $OUT_DIR --target-layer 6"
