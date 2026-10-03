#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=04ed74fbc2912af58d65c487ce9947ea5cc71dc1
STAGE=tests
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/hamster3d_full1_validation_v1/$REV"
PY="$P/.venv-hamster3d/bin/python"
RUNNER="$P/research_v2/incoming/hamster_full1_runner_04ed74f.py"
export FROZEN_HAMSTER_FULL1_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$FROZEN_HAMSTER_FULL1_WRAPPER" == "$P/research_v2/incoming/hamster_full1_${STAGE}_${REV:0:7}.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == d13c0b4f35a7d8d07be8ee780ae52bfe04cb4bd4474210e0f6f24cab1d38ee2e ]] || exit 2
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE.lock" && ! -e "$OUT/${STAGE}_identity.json" && ! -e "$OUT/${STAGE}_receipt.json" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 0 "$PY" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$PY" "$RUNNER" --stage "$STAGE"
