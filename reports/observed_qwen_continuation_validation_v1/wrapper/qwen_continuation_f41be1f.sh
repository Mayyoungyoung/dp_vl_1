#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=f41be1ff1a35b0eab3d36edf4ca23197359f4731
SRC="$P/research_v2/releases/$REV"
STAGE="${1:?tests or train or fixed-last-train}"
ARM="${2:-none}"
MODE="${3:-full}"
BASE="$P/runs/observed_two_row_lora_continuation_v1"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
PY="$P/.venv-qwen/bin/python"
if [[ "$STAGE" == tests ]]; then
 OUT="$P/runs/observed_qwen_continuation_validation_v1/$REV"
 RUN=tests
 ARGS=(-m pytest -q -p no:cacheprovider --junitxml="$OUT/pytest.xml" tests/test_observed_qwen_continuation.py tests/test_qwen_prefix_corpus.py tests/test_qwen_prefix_replay.py)
else
 [[ "$ARM" == frozen || "$ARM" == lora ]] || exit 2
 export CUDA_VISIBLE_DEVICES=1
 OUT="$BASE/job_records"
 RUN="${ARM}_${STAGE}_${MODE}"
 TARGET="$BASE/$ARM"
 ARGS=(-m scripts.train_observed_two_row_lora_continuation --stage "$STAGE" --arm "$ARM" --config "$SRC/configs/observed_two_row_lora_continuation_v1.json" --model "$P/data/qwen3-vl-2b-instruct-89644892" --data "$P/data/observation_two_row_composite108_v1" --head "$P/runs/observed_two_row_composite108_v1/peak_seed0/last.pt" --prefix-corpus "$P/data/observation_two_row_composite108_v1/qwen_prefix_corpus" --draw-plan "$BASE/shared_draw_plan.json" --quality-audit "$P/runs/observed_two_row_composite108_preparation_v1/train_quality.json")
 case "$STAGE" in
  train)
   case "$MODE" in
    pause2) ARGS+=(--stop-after 2);;
    resume) ARGS+=(--resume);;
    full) ;;
    *) exit 2;;
   esac
   ;;
  fixed-last-train)
   [[ "$MODE" == full ]] || exit 2
   ARGS+=(--train-run "$TARGET")
   TARGET="$BASE/${ARM}_fixed_last_train"
   ;;
  *) exit 2;;
 esac
 ARGS+=(--output "$TARGET")
fi
[[ ! -e "$OUT/$RUN.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$RUN" --resume-strategy none -- "$PY" "${ARGS[@]}"
