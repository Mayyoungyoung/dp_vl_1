#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Immutable source commit required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT="$P/runs/observed_two_row_bound_controls_validation_v1/$REVISION"
[[ ! -e "$OUT/targeted_tests.status.json" ]] || exit 2
bash -n scripts/launch_observed_two_row_convergence64_v1.sh scripts/launch_observed_two_row_spatial_penalty_v1.sh
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id targeted_tests --resume-strategy none -- \
  "$P/.venv/bin/python" -m pytest -q tests/test_two_row_convergence64.py tests/test_two_row_convergence.py \
  tests/test_observation_spatial_penalty_astar.py tests/test_two_row_astar_control.py \
  tests/test_two_row_observation_training.py tests/test_two_row_last_train.py
