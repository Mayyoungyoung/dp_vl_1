#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Immutable commit required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
OUT="$P/runs/observed_two_row_cosine_validation_v1/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$P/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
assert len(os.sched_getaffinity(0)) == 1
assert Path.cwd().name == os.environ['CODE_COMMIT']
PY
[[ ! -e "$OUT/targeted_tests.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id targeted_tests --resume-strategy none -- \
  "$P/.venv/bin/python" -m pytest -q -p no:cacheprovider \
  --junitxml="$OUT/pytest.xml" \
  tests/test_two_row_cosine.py tests/test_two_row_convergence64.py tests/test_two_row_convergence.py \
  tests/test_two_row_observation_training.py tests/test_two_row_last_train.py
