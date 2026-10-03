#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=6e0203ba1335f9fa9975c523c657959f8bc9ab60
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_diffusion_validation_v1/$REV"
RUNNER="$P/research_v2/incoming/observed_diffusion_validation_runner_6e0203b.py"
export DIFFUSION_VALIDATION_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == a1bca2711173021757096003b84e99b10c61b1dad43e03b1c65d44713ed4de62 ]] || exit 2
[[ ! -e "$OUT/tests.status.json" && ! -e "$OUT/tests.lock" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id tests --resume-strategy none -- \
  "$P/.venv/bin/python" "$RUNNER" "$SRC" "$OUT"
