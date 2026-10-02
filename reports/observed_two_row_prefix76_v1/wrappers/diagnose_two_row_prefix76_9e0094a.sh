#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
SRC=$P/research_v2/releases/9e0094aff152d48cd34eadd24e554a05f0cf4a0f
cd "$SRC"
export CODE_COMMIT=9e0094aff152d48cd34eadd24e554a05f0cf4a0f PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_prefix76_diagnosis_v1" --run-id analysis --resume-strategy fresh-output -- "$P/.venv/bin/python" -m scripts.analyze_two_row_ordinary --run "$P/runs/observed_two_row_prefix76_v1/peak_seed0" --data "$P/data/observation_two_row_prefix76_v1" --output "$P/runs/observed_two_row_prefix76_diagnosis_v1/analysis"
exec "$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_prefix76_last_train_v1" --run-id analysis --resume-strategy fresh-output -- "$P/.venv/bin/python" -m scripts.audit_two_row_last_train --run "$P/runs/observed_two_row_prefix76_v1/peak_seed0" --data "$P/data/observation_two_row_prefix76_v1" --output "$P/runs/observed_two_row_prefix76_last_train_v1/analysis"
