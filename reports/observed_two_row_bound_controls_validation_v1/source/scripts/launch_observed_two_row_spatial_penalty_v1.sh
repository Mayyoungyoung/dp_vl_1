#!/usr/bin/env bash
# Two separately reviewed stages; no automatic replay of a candidate request.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Immutable source commit required}"
STAGE="${2:?train_preflight or dev required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$STAGE" == train_preflight || "$STAGE" == dev ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
OUT="$P/runs/observed_two_row_spatial_penalty_v1"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$P/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
assert len(os.sched_getaffinity(0))==1, 'One authorized CPU required'
assert Path.cwd().name==os.environ['CODE_COMMIT']
PY
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE" ]] || { echo 'Preserve existing stage artifacts; no automatic request replay.' >&2; exit 2; }
EXTRA=()
if [[ "$STAGE" == dev ]]; then
  [[ -f "$OUT/train_preflight/report.json" ]] || exit 2
  EXTRA+=(--preflight "$OUT/train_preflight/report.json")
fi
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$P/.venv/bin/python" -m scripts.observation_spatial_penalty_astar \
  --data "$P/data/observation_two_row_prefix44_v1" --output "$OUT/$STAGE" \
  --config "$SRC/configs/observed_two_row_spatial_penalty_v1.json" --stage "$STAGE" "${EXTRA[@]}"
