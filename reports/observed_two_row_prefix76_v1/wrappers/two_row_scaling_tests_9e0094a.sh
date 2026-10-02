#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
SRC=$P/research_v2/releases/9e0094aff152d48cd34eadd24e554a05f0cf4a0f
cd "$SRC"
export CODE_COMMIT=9e0094aff152d48cd34eadd24e554a05f0cf4a0f PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observation_two_row_scaling_validation_v1" --run-id targeted_tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest -q tests/test_two_row_observation_export.py tests/test_two_row_train_quality.py tests/test_two_row_observation_training.py tests/test_two_row_ordinary_analysis.py tests/test_two_row_formal_train_analysis.py tests/test_observed_two_row_evaluation.py tests/test_two_row_formal_collection.py tests/test_two_row_formal_registration.py tests/test_observed_geometry.py tests/test_observed_grounding_target_training.py tests/test_two_row_scaling.py tests/test_two_row_last_train.py tests/test_two_row_online_evaluation.py
