#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c
SRC="$P/research_v2/releases/$REV"
STAGE="${1:?tests or quality or cache or train or fixed-last-train}"
PREP="$P/runs/observed_two_row_composite108_preparation_v1"
TRAIN="$P/runs/observed_two_row_composite108_v1/peak_seed0"
DATA="$P/data/observation_two_row_composite108_v1"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
PY="$P/.venv/bin/python"
OUT="$PREP"
ARGS=(-m scripts.run_observed_two_row_composite --stage "$STAGE" --data "$DATA" --policy "$SRC/configs/observed_two_row_composite108_training_v1.json")
case "$STAGE" in
 tests)
  OUT="$P/runs/observed_two_row_composite108_validation_v1/$REV"
  ARGS=(-m pytest -q -p no:cacheprovider --junitxml="$OUT/pytest.xml" tests/test_two_row_composite_pipeline.py tests/test_two_row_composite_training_torch.py tests/test_two_row_train_quality.py tests/test_two_row_observation_training.py tests/test_two_row_composite_export.py)
  ;;
 quality) ARGS+=(--output "$PREP/train_quality.json") ;;
 cache)
  export CUDA_VISIBLE_DEVICES=1
  PY="$P/.venv-qwen/bin/python"
  ARGS+=(--quality-audit "$PREP/train_quality.json" --model "$P/data/qwen3-vl-2b-instruct-89644892" --output "$PREP/cache")
  ;;
 train)
  export CUDA_VISIBLE_DEVICES=1
  OUT="$P/runs/observed_two_row_composite108_v1"
  ARGS+=(--quality-audit "$PREP/train_quality.json" --output "$TRAIN")
  ;;
 fixed-last-train)
  OUT="$P/runs/observed_two_row_composite108_v1"
  ARGS+=(--quality-audit "$PREP/train_quality.json" --model-folder "$TRAIN" --output "$OUT/fixed_last_train")
  ;;
 *) exit 2;;
esac
[[ ! -e "$OUT/$STAGE.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- "$PY" "${ARGS[@]}"
