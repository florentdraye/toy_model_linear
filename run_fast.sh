#!/usr/bin/env bash
# Fast preset: 10^6 paths, WD=1.0, batch=8192, 30 epochs (~58s on H100).
# Train and test track together; model learns graph structure rather than memorizing.
# Override on the CLI: ./run_fast.sh --n-epochs 60 --weight-decay 3.0
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)-fast}"
mkdir -p "$OUT_DIR"

python -u train.py \
  --n-layers 7 \
  --nodes-per-layer 1 100 100 100 100 100 100 \
  --edges-per-node 10 \
  --lr 3e-3 \
  --weight-decay 0.0 \
  --batch-size 1024 \
  --n-epochs 60 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"

