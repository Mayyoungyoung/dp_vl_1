#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
STAGE="$1"
case "$STAGE" in tests|inspect-parent|train_a|fixed_a|train_b|fixed_b|train_c|fixed_c) ;; *) exit 2;; esac
P=/home/wzy/dpvlm/route_set_v1
REV=ece16ba41daab9cfcb9422f0014e640a1c934c7b
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_ordered_relation_continuation_v1"
RUNNER="$P/research_v2/incoming/ordered_training_runner_ece16ba.py"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 34300c36b41f986b0ca83b46ac6d18aa44ca6cf64b3e2a2f06d4f000dba155bc ]] || exit 2
[[ ! -e "$OUT/job_records/$STAGE.status.json" && ! -e "$OUT/job_records/$STAGE.lock" && ! -e "$OUT/${STAGE}_identity.json" && ! -e "$OUT/${STAGE}_receipt.json" ]] || exit 2
cd "$SRC"
export ORDERED_TRAINING_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
case "$STAGE" in
 tests|inspect-parent) export CUDA_VISIBLE_DEVICES=''; unset RESEARCH_GPU_UUID;;
 *) export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab;;
esac
exec taskset -c 0 "$P/.venv/bin/python" -m scripts.record_job --output "$OUT/job_records" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" "$RUNNER" "$STAGE"
