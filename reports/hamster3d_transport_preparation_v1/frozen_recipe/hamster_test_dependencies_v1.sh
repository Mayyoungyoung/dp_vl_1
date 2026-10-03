#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=cbcd8129c967db6f0995a824754752cc7a6f0f5b
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/hamster3d_test_dependencies_v1"
RUNNER="$P/research_v2/incoming/hamster_test_dependencies_v1.py"
PY="$P/.venv-hamster3d/bin/python"
export FROZEN_HAMSTER_DEPENDENCY_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$FROZEN_HAMSTER_DEPENDENCY_WRAPPER" == "$P/research_v2/incoming/hamster_test_dependencies_v1.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 0b483d76d50d78e5983e02bbc58da34ecc1aadb73fb125ff5a3de4139f38fba5 ]] || exit 2
[[ ! -e "$OUT/install.status.json" && ! -e "$OUT/install.lock" && ! -e "$OUT/install_receipt.json" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 0 "$PY" -m scripts.record_job --output "$OUT" --run-id install --resume-strategy none -- \
 "$PY" "$RUNNER"
