#!/usr/bin/env bash
set -euo pipefail
cd /lustre/home/fdraye/projects/toy_model_linear
export OMP_NUM_THREADS=4
seed="$1"
latent="$2"
python_bin=/lustre/home/fdraye/projects/granularity/.venv/bin/python
run="/fast/fdraye/toy_model_linear/emergence_optimized16_source_s${seed}"
out="/fast/fdraye/toy_model_linear/context_variation_localparent_dense_s${seed}/latent_${latent}"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
exec "$python_bin" -u measure_context_transfer.py --run "$run" --out-dir "$out" \
  --latent "$latent" --variation-only --time-every 100 --pair-mode local-parent
