#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
STAGE="$1"
case "$STAGE" in tests|inspect-parent|uniform|min_snr_5|compare) ;; *) exit 2;; esac
P=/home/wzy/dpvlm/route_set_v1
REV=f72e1aa11a37df2001877035fbfde9a4525c0aa0
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_diffusion_minsnr_probe_v2"
RUNNER="$P/research_v2/incoming/minsnr_runner_f72e1aa.py"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 332e277aefa2962ee4c261c503219a08be25a7c81e060f889c740c53d4fc4d34 ]] || exit 2
[[ ! -e "$OUT/job_records/$STAGE.status.json" && ! -e "$OUT/job_records/$STAGE.lock" && ! -e "$OUT/${STAGE}_receipt.json" && ! -e "$OUT/${STAGE}_identity.json" ]] || exit 2
cd "$SRC"
export MINSNR_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
case "$STAGE" in
 tests|inspect-parent|compare) export CUDA_VISIBLE_DEVICES=''; unset RESEARCH_GPU_UUID;;
 uniform|min_snr_5) export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab;;
esac
exec taskset -c 0 "$P/.venv/bin/python" -m scripts.record_job --output "$OUT/job_records" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" "$RUNNER" "$STAGE"
