#!/usr/bin/env bash
set -euo pipefail
cd /lustre/home/fdraye/projects/toy_model_linear
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/fast/fdraye/toy_model_linear/matplotlib
mkdir -p "$MPLCONFIGDIR"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
seed="$1"
shift
exec /lustre/home/fdraye/projects/granularity/.venv/bin/python -u train_emergence.py \
  --out-dir "/fast/fdraye/toy_model_linear/dom_multiseed_s${seed}" --seed "$seed" \
  --d-model 512 --n-blocks 6 --steps 20000 --eval-every 100 \
  --num-latents 32 --latent-ranks 1 4 6 8 11 13 15 18 20 22 25 29 32 36 39 43 46 50 53 57 60 64 67 71 74 78 81 85 88 92 95 99 \
  --site suffix --direction-method uniform-reference-mean --steer-depth 1 \
  --frequency-ratio 10 --weight-decay 0.1 --lr 0.001 --batch-size 2048 \
  --mean-batch 8192 --best-locations --save-models --model-save-every 1000 "$@"
