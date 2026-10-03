#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
STAGE="$1"
case "$STAGE" in tests|bench) ;; *) exit 2;; esac
P=/home/wzy/dpvlm/route_set_v1
REV=e69ab8a134fee4eb8510c1dd52fb90e68a9e8a1c
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_ordered_relation_microbenchmark_v1"
RUNNER="$P/research_v2/incoming/ordered_relation_runner_e69ab8a.py"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 99cdb2e5a43e2effbeb46e92690e1666107b65bc2244fd0a24b5587dc1abd19e ]] || exit 2
[[ ! -e "$OUT/job_records/$STAGE.status.json" && ! -e "$OUT/job_records/$STAGE.lock" && ! -e "$OUT/${STAGE}_receipt.json" && ! -e "$OUT/${STAGE}_identity.json" && ! -e "$OUT/active.lock" ]] || exit 2
cd "$SRC"
export ORDERED_RELATION_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
if [[ "$STAGE" == tests ]]; then
 export CUDA_VISIBLE_DEVICES=''; unset RESEARCH_GPU_UUID
else
 export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
fi
exec taskset -c 2 "$P/.venv/bin/python" -m scripts.record_job --output "$OUT/job_records" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" "$RUNNER" "$STAGE"
