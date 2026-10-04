#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CODE_COMMIT="$(basename "$PWD")"
exec taskset -c 0-3 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m scripts.goal_job "$@"
