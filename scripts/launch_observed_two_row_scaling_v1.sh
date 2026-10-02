#!/usr/bin/env bash
# Independent stages: CPU prepare, root review, then a separately launched GPU stage.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Reviewed immutable source commit required}"
TRAIN_PARENTS="${2:?Fixed TRAIN parent count 32 or 64 required}"
STAGE="${3:?Separate prepare or gpu stage required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$TRAIN_PARENTS" == 32 || "$TRAIN_PARENTS" == 64 ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONUNBUFFERED=1 PYTHONPATH="$SRC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "$STAGE" in
  prepare) export CUDA_VISIBLE_DEVICES='' ;;
  gpu) export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab ;;
  *) exit 2 ;;
esac
LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
exec "$P/.venv/bin/python" -m scripts.run_observed_two_row_scaling \
  --train-parents "$TRAIN_PARENTS" --stage "$STAGE" --launcher "$LAUNCHER"
