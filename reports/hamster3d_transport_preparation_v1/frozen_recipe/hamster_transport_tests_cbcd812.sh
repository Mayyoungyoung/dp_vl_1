#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=cbcd8129c967db6f0995a824754752cc7a6f0f5b
STAGE=tests
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/hamster3d_transport_validation_v1/$REV"
PY="$P/.venv-hamster3d/bin/python"
RUNNER="$P/research_v2/incoming/hamster_transport_runner_cbcd812.py"
export FROZEN_HAMSTER_TRANSPORT_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$FROZEN_HAMSTER_TRANSPORT_WRAPPER" == "$P/research_v2/incoming/hamster_transport_${STAGE}_${REV:0:7}.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 49558f81fb7a3a24bbac7835cb24cd89c02ec9ba202252b8e831d7de3d774c58 ]] || exit 2
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE.lock" && ! -e "$OUT/${STAGE}_identity.json" && ! -e "$OUT/${STAGE}_receipt.json" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 0 "$PY" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$PY" "$RUNNER" --stage "$STAGE"
