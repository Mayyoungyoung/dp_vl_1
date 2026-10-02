#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Immutable source commit required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT="$P/runs/observed_two_row_cross_goal_opportunity_v1"
[[ ! -e "$OUT/analysis.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id analysis --resume-strategy fresh-output -- \
 "$P/.venv/bin/python" -m scripts.audit_two_row_cross_goal_correspondence \
 --config "$SRC/configs/two_row_cross_goal_opportunity_v1.json" --output "$OUT/analysis"
