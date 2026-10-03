#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=1b0348ef48393a2d98113956575e99388c40d4a6
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/hamster3d_preparation_validation_v1/$REV"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
[[ ! -e "$OUT/tests.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id tests --resume-strategy none -- "$P/.venv-qwen/bin/python" -m pytest -q -p no:cacheprovider --junitxml="$OUT/pytest.xml" tests/test_hamster3d_preparation.py
