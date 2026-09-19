#!/usr/bin/env bash
set -euo pipefail
cd /lustre/home/fdraye/projects/toy_model_linear
export OMP_NUM_THREADS=4
seed="$1";python_bin=/lustre/home/fdraye/projects/granularity/.venv/bin/python
exec "$python_bin" -u measure_count_subset_scatter.py \
  --run "/fast/fdraye/toy_model_linear/emergence_optimized16_source_s${seed}" \
  --out-dir "/fast/fdraye/toy_model_linear/count_subset_scatter_l78_s${seed}"
