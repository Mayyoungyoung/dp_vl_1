#!/usr/bin/env bash
set -euo pipefail
# Prepared only. Root must verify all128 mechanical closure and authorize launch.
P=/home/wzy/dpvlm/route_set_v1
REV=1a3eef1fb12d55e98d4d188a091ea40ea62c0a02
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_two_row_extension288_quality_v1"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
[[ ! -e "$OUT/train128.status.json" && ! -e "$OUT/train128" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id train128 --resume-strategy none -- \
 "$P/.venv/bin/python" -m scripts.analyze_two_row_extension_train --corpus "$P/data/observed_two_row_extension288_v1" --train-parents 128 --output "$OUT/train128"
