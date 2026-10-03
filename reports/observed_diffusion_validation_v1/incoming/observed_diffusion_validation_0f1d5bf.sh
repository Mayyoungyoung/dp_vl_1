#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=0f1d5bfbd4ff788a6ea311339adfe77449db9634
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_diffusion_validation_v1/$REV"
RUNNER="$P/research_v2/incoming/observed_diffusion_validation_runner_0f1d5bf.py"
export DIFFUSION_VALIDATION_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 08c5bf34e173a7d2772b8a7b408d95edb0b32296c0a359ef6c3771213154f2a0 ]] || exit 2
[[ ! -e "$OUT/tests.status.json" && ! -e "$OUT/tests.lock" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id tests --resume-strategy none -- \
  "$P/.venv/bin/python" "$RUNNER" "$SRC" "$OUT"
