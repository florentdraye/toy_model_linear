#!/usr/bin/env bash
set -euo pipefail
cd /lustre/home/fdraye/projects/toy_model_linear
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/fast/fdraye/toy_model_linear/matplotlib
mkdir -p "$MPLCONFIGDIR"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

seed="$1"
source_dir="/fast/fdraye/toy_model_linear/emergence_optimized16_source_s${seed}"
refined_dir="/fast/fdraye/toy_model_linear/emergence_optimized16_refined_s${seed}"
python_bin=/lustre/home/fdraye/projects/granularity/.venv/bin/python

"$python_bin" -u train_emergence.py \
  --out-dir "$source_dir" --seed "$seed" --steps 10000 --eval-every 100 \
  --d-model 128 --n-blocks 6 --site suffix --steer-depth 1 \
  --direction-method optimized --direction-steps 400 --direction-lr 0.03 \
  --frequency-ratio 10 --weight-decay 0.1 --lr 0.001 --batch-size 2048 \
  --num-latents 16 --latent-ranks 29 34 38 43 48 52 57 62 66 71 76 80 85 90 94 99 \
  --save-models

exec "$python_bin" -u refine_emergence.py "$source_dir" \
  --out-dir "$refined_dir" --steps 400
