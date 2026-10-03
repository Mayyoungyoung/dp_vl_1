#!/usr/bin/env bash
set -euo pipefail
# PREPARED ONLY: root must confirm both train256 shards completed and all256 closures.
P=/home/wzy/dpvlm/route_set_v1
REV=1a3eef1fb12d55e98d4d188a091ea40ea62c0a02
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_two_row_extension288_quality_v1"
[[ "$(sha256sum "$SRC/scripts/analyze_two_row_extension_train.py" | awk '{print $1}')" == 758235704e344b0f34cce414b0bacb8f28027ef6b1eaef6f96e21279838bfcba ]] || exit 2
[[ ! -e "$OUT/train256.status.json" && ! -e "$OUT/train256" && ! -e "$OUT/train256.lock" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec taskset -c 1 "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id train256 --resume-strategy none -- \
 "$P/.venv/bin/python" -m scripts.analyze_two_row_extension_train --corpus "$P/data/observed_two_row_extension288_v1" --train-parents 256 --output "$OUT/train256"
