#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Reviewed immutable source commit required}"
COUNT="${2:?Fixed registered TRAIN parent count 32 or 64 required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
case "$COUNT" in
  32) PREFIX=44 ;;
  64) PREFIX=76 ;;
  *) exit 2 ;;
esac
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$P/.venv/bin/python" -m scripts.record_job \
  --output "$P/runs/observation_two_row_astar_prefix${PREFIX}_v1" --run-id dev_model --resume-strategy fresh-output -- \
  "$P/.venv/bin/python" -m scripts.evaluate_two_row_astar_v2 \
  --data "$P/data/observation_two_row_prefix${PREFIX}_v1" \
  --output "$P/runs/observation_two_row_astar_prefix${PREFIX}_v1/dev_model"
