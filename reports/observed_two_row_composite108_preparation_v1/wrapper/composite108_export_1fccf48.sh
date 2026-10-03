#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=1fccf4898bfa17233f92e20476adeb8db12c6b60
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_two_row_composite108_preparation_v1"
STAGE="${1:?readiness or export}"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ARGS=(--old-source "$P/data/observed_two_row_formal116_v1" --extension-source "$P/data/observed_two_row_extension288_v1" --old-export "$P/data/observation_two_row_prefix76_v1" --selection "$SRC/configs/observed_two_row_composite108_v1.json")
if [[ "$STAGE" == readiness ]]; then ARGS+=(--readiness-only)
elif [[ "$STAGE" == export ]]; then ARGS+=(--output "$P/data/observation_two_row_composite108_v1")
else exit 2; fi
[[ ! -e "$OUT/$STAGE.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- "$P/.venv/bin/python" -m scripts.export_two_row_composite_observations "${ARGS[@]}"
