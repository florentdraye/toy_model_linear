#!/usr/bin/env bash
set -euo pipefail
cd /lustre/home/fdraye/projects/toy_model_linear
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
exec /lustre/home/fdraye/projects/granularity/.venv/bin/python -u audit_dom_multiseed.py \
  /fast/fdraye/toy_model_linear/dom_multiseed_s{46..55} "$@"
