#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 CODE_COMMIT="$(basename "$PWD")" SELECTIVE_BODY_CPU_SET=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C LANG=C
exec taskset -c 2 /home/wzy/dpvlm/route_set_v1/.venv-sim/bin/python -m research_selective_repair_v1.event_execution_receipt "$@"
