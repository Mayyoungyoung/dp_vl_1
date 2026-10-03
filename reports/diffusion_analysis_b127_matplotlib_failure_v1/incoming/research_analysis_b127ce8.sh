#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=b127ce8514a63917cc3f1b7278db8f65289ba985
STAGE=analysis
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_two_row_diffusion_analysis_v1"
PY="$P/.venv-qwen/bin/python"
RUNNER="$P/research_v2/incoming/research_stage_runner_b127ce8.py"
export FROZEN_RESEARCH_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$FROZEN_RESEARCH_WRAPPER" == "$P/research_v2/incoming/research_${STAGE}_${REV:0:7}.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 5c2c70afd03c0831fd7965ef77d6534bb92e5b94689a322e098fad3c78538fb6 ]] || exit 2
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE.lock" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 0 "$PY" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$PY" "$RUNNER" --stage "$STAGE"
