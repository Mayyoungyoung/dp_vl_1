#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LC_ALL=C LANG=C
exec taskset -c 3 /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_selective_repair_v1.cpu_test_receipt --name "${1:-body_prefix_cpu_tests_v1}"
