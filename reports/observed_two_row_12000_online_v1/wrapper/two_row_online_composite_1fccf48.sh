#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=1fccf4898bfa17233f92e20476adeb8db12c6b60
SRC="$P/research_v2/releases/$REV"
STAGE="${1:?tests or preflight_constant or preflight_no_direct or constant or no_direct}"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
OUT="$P/runs/observed_two_row_12000_online_v1"
PY="$P/.venv/bin/python"
if [[ "$STAGE" == tests ]]; then
  OUT="$P/runs/observed_two_row_online_composite_validation_v1/$REV"
  ARGS=(-m pytest -q -p no:cacheprovider --junitxml="$OUT/pytest.xml" tests/test_two_row_12000_online.py tests/test_two_row_online_evaluation.py tests/test_two_row_composite_export.py tests/test_two_row_observation_export.py)
elif [[ "$STAGE" == preflight_constant || "$STAGE" == preflight_no_direct ]]; then
  ARM="${STAGE#preflight_}"
  ARGS=(-m scripts.evaluate_two_row_12000_online --arm "$ARM" --project "$P" --model "$P/data/qwen3-vl-2b-instruct-89644892" --output "$OUT/$STAGE" --validate-only)
elif [[ "$STAGE" == constant || "$STAGE" == no_direct ]]; then
  export CUDA_VISIBLE_DEVICES=1
  [[ "$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader)" == "$RESEARCH_GPU_UUID" ]] || exit 2
  PY="$P/.venv-qwen/bin/python"
  ARGS=(-m scripts.evaluate_two_row_12000_online --arm "$STAGE" --project "$P" --model "$P/data/qwen3-vl-2b-instruct-89644892" --output "$OUT/$STAGE")
else exit 2; fi
[[ ! -e "$OUT/$STAGE.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- "$PY" "${ARGS[@]}"
