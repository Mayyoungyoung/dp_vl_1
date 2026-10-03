#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
STAGE="$1"
case "$STAGE" in tests|metadata|pause2|resume) ;; *) exit 2;; esac
P=/home/wzy/dpvlm/route_set_v1
REV=443ea311d53cf33ff7ab4fe3d533bb61deee4726
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/budget_conditioned_regression_v1"
RUNNER="$P/research_v2/incoming/budget_runner_443ea31.py"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 231b0bb506b951af975b97fd5a2dbaeb70b96d8dde10e92d6c0c24e59abde547 ]] || exit 2
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE.lock" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
if [[ "$STAGE" == tests ]]; then export CUDA_VISIBLE_DEVICES=''; unset RESEARCH_GPU_UUID; fi
exec taskset -c 0 "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" "$RUNNER" "$STAGE"
