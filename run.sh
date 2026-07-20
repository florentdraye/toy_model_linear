#!/usr/bin/env bash
# Train and stream output to your terminal in real time.
# Edit the args below or override on the CLI: ./run.sh --n-epochs 30 --weight-decay 3.0
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR"

# `-u` = unbuffered stdout so prints appear line-by-line
# `tee` so the log is also saved for later inspection
python -u train.py \
  --n-layers 5 \
  --nodes-per-layer 1 100 100 100 100 \
  --edges-per-node 10 \
  --weight-decay 0.0001 \
  --batch-size 256 \
  --n-epochs 30 \
  --out-dir "$OUT_DIR" \
  "$@" \
  2>&1 | tee "$OUT_DIR/log.txt"
