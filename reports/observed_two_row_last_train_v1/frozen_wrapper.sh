#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
SRC=$P/research_v2/releases/743d9b269c27054b05c48cc5bab15017ab3fae55
cd "$SRC"
export CODE_COMMIT=743d9b269c27054b05c48cc5bab15017ab3fae55 PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_last_train_v1" --run-id targeted_tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_two_row_last_train.py -q
exec "$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_last_train_v1" --run-id analysis --resume-strategy none -- "$P/.venv/bin/python" -m scripts.audit_two_row_last_train --run "$P/runs/observed_two_row_prefix28_v1/peak_seed0" --data "$P/data/observation_two_row_prefix28_v1" --output "$P/runs/observed_two_row_last_train_v1/analysis"
