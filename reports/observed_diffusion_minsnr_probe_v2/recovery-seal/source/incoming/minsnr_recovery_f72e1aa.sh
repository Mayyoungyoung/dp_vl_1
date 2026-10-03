#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || exit 2
STAGE="$1"
case "$STAGE" in seal-uniform|min_snr_5|compare) ;; *) exit 2;; esac
P=/home/wzy/dpvlm/route_set_v1
REV=f72e1aa11a37df2001877035fbfde9a4525c0aa0
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_diffusion_minsnr_probe_v2_recovery_v1"
RUNNER="$P/research_v2/incoming/minsnr_recovery_f72e1aa.py"
MANIFEST="$P/research_v2/incoming/minsnr_recovery_f72e1aa_manifest.json"
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 221c98c22aa886a2fce871753ded817ec44ba916e811cd9a9d133ac5401c89de ]] || exit 2
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == 7ac551936241dd3d48bd4c4560439501cf4dc9b70f4691e124145a0c1c0af33f ]] || exit 2
[[ ! -e "$OUT/job_records/$STAGE.status.json" && ! -e "$OUT/job_records/$STAGE.lock" && ! -e "$OUT/${STAGE}_receipt.json" && ! -e "$OUT/${STAGE}_identity.json" ]] || exit 2
cd "$SRC"
export MINSNR_RECOVERY_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
case "$STAGE" in
 seal-uniform|compare) export CUDA_VISIBLE_DEVICES=''; unset RESEARCH_GPU_UUID;;
 min_snr_5) export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab;;
esac
exec taskset -c 0 "$P/.venv/bin/python" -m scripts.record_job --output "$OUT/job_records" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" "$RUNNER" "$STAGE"
