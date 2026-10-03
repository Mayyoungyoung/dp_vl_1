#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 0 ]] || exit 2
P=/home/wzy/dpvlm/route_set_v1
REV=eba09945ecade4bdb9a5b4ab123a92da898283ed
STAGE=pilot12
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_layout_variation_train12_v1"
PY="$P/.venv/bin/python"
INCOMING="$P/research_v2/incoming"
RUNNER="$INCOMING/layout_variation_pilot12_runner_eba0994.py"
export FROZEN_LAYOUT_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$FROZEN_LAYOUT_WRAPPER" == "$INCOMING/layout_variation_pilot12_eba0994.sh" ]] || exit 2
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == 118b12310d78d82aab346cd31a0aa11c320719618e75f60ee5517672ab1ab49f ]] || exit 2
[[ "$(sha256sum "$INCOMING/layout_variation_runner_eba0994.py" | awk '{print $1}')" == 6fbaeca615247e6bd76ca5d7fff10bace0265daf2c0d580d98305bce8f72e586 ]] || exit 2
[[ "$(sha256sum "$INCOMING/layout_variation_pilot12_recipe_eba0994.json" | awk '{print $1}')" == 421c74ba021883bee16bcfb4fbf331715410a513bbd6ddb5d7ac9424057247cf ]] || exit 2
[[ ! -e "$OUT/$STAGE.status.json" && ! -e "$OUT/$STAGE.lock" && ! -e "$OUT/recipe.lock" && ! -e "$OUT/$STAGE" && ! -e "$OUT/${STAGE}_recipe" && ! -e "$OUT/${STAGE}_receipt.json" && ! -e "$OUT/${STAGE}_identity.json" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
exec taskset -c 1 "$PY" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- \
  "$PY" "$RUNNER"
