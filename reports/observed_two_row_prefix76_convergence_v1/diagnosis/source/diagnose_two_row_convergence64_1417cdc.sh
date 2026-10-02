#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
SRC=$P/research_v2/releases/1417cdc802674a00e70d4ced3181c6a6ede2a99e
cd "$SRC"
export CODE_COMMIT=1417cdc802674a00e70d4ced3181c6a6ede2a99e PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_convergence64_diagnosis_v1" --run-id analysis --resume-strategy fresh-output -- \
 "$P/.venv/bin/python" -m scripts.analyze_two_row_ordinary --data "$P/data/observation_two_row_prefix76_v1" \
 --run "$P/runs/observed_two_row_prefix76_convergence_v1/peak_seed0" \
 --output "$P/runs/observed_two_row_convergence64_diagnosis_v1/analysis"
