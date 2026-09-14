#!/usr/bin/env bash
set -euo pipefail
cd /lustre/home/fdraye/projects/toy_model_linear
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR=/fast/fdraye/toy_model_linear/matplotlib
mkdir -p "$MPLCONFIGDIR"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
exec /lustre/home/fdraye/projects/granularity/.venv/bin/python -u measure_batch_kernel.py "$@"
