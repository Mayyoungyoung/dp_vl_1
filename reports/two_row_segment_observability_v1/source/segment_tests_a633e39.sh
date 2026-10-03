#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=a633e39204623cf1760f369e70d1c3816074b4d1
STAGE=tests
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/two_row_segment_observability_validation_v1/$REV"
PY="$P/.venv-qwen/bin/python"
RUNNER="$P/research_v2/incoming/segment_stage_runner_a633e39.py"
export FROZEN_SEGMENT_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$FROZEN_SEGMENT_WRAPPER" == "$P/research_v2/incoming/segment_${STAGE}_${REV:0:7}.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 28dca9c94f7e92a0487171cc72b0b22fa10e748e7616fce62a36caaca23e1d5b ]] || exit 2
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE.lock" && ! -e "$OUT/$STAGE.log" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 1 "$PY" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$PY" "$RUNNER" --stage "$STAGE"
