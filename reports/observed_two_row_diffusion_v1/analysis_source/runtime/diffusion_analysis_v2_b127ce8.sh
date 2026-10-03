#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=b127ce8514a63917cc3f1b7278db8f65289ba985
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_two_row_diffusion_analysis_v2"
PY="$P/.venv/bin/python"
RUNNER="$P/research_v2/incoming/diffusion_analysis_v2_runner_b127ce8.py"
export DIFFUSION_ANALYSIS_V2_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$DIFFUSION_ANALYSIS_V2_WRAPPER" == "$P/research_v2/incoming/diffusion_analysis_v2_b127ce8.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 799afa8c16aa408f08a69280ec2c681d509cd1c349025f7fc050ccf95e326dfc ]] || exit 2
[[ ! -e "$OUT/analysis.status.json" && ! -e "$OUT/analysis.lock" && ! -e "$OUT/analysis" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 0 "$PY" -m scripts.record_job --output "$OUT" --run-id analysis --resume-strategy none -- \
  "$PY" "$RUNNER"
