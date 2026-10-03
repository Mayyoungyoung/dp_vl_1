#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=a3daf0da0c7a30279d38e9a4a18ee93b989911d4
SRC="$P/research_v2/releases/$REV"
STAGE="${1:?self-test or analyze}"
BASE="$P/runs/observed_two_row_lora_continuation_v1"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
OUT="$P/runs/observed_two_row_lora_continuation_analysis_v1"
case "$STAGE" in
 self-test) ARGS=(-m scripts.analyze_observed_qwen_continuation --self-test);;
 analyze)
  ARGS=(-m scripts.analyze_observed_qwen_continuation --runs "$BASE" --initial-run "$P/runs/observed_two_row_composite108_v1" --job-records "$BASE/job_records" --draw-plan "$BASE/shared_draw_plan.json" --training-source "$P/research_v2/releases/f41be1ff1a35b0eab3d36edf4ca23197359f4731" --prefix-corpus "$P/data/observation_two_row_composite108_v1/qwen_prefix_corpus" --prefix-job-status "$P/runs/observed_qwen_prefix_corpus_v1/cache.status.json" --output "$OUT/analysis")
  ;;
 *) exit 2;;
esac
[[ ! -e "$OUT/$STAGE.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" "${ARGS[@]}"
