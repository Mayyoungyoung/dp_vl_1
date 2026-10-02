#!/usr/bin/env bash
# Independent fresh stages. Never chain TRAIN/DEV or retry a candidate request.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Immutable source commit required}"
STAGE="${2:?tests, train_preflight or dev required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$STAGE" == tests || "$STAGE" == train_preflight || "$STAGE" == dev ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
OUT="$P/runs/observed_two_row_native_astar_100k_v1"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$P/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
assert len(os.sched_getaffinity(0)) == 1, 'One authorized CPU required'
assert Path.cwd().name == os.environ['CODE_COMMIT']
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
PY
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE" ]] || {
  echo 'Preserve existing stage artifacts; no automatic replay.' >&2; exit 2;
}
if [[ "$STAGE" != tests ]]; then
  [[ -f "$OUT/tests/report.json" && -f "$OUT/tests.status.json" ]] || exit 2
fi
if [[ "$STAGE" == dev ]]; then
  [[ -f "$OUT/train_preflight/report.json" && -f "$OUT/train_preflight.status.json" ]] || exit 2
fi
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$P/.venv/bin/python" -m scripts.run_observed_native_astar_100k --source-commit "$REVISION" --stage "$STAGE"
