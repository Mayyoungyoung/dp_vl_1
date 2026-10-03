#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=eddfacaddab2d12c67f5a56fd775de173c05f9b2
SRC="$P/research_v2/releases/$REV"
STAGE="${1:?tests or probe}"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES='' RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
PY="$P/.venv-qwen/bin/python"
case "$STAGE" in
 tests)
  OUT="$P/runs/observed_qwen_prefix_replay_validation_v1/$REV"
  ARGS=(-m pytest -q -p no:cacheprovider --junitxml="$OUT/pytest.xml" tests/test_qwen_prefix_replay.py tests/test_observed_geometry.py tests/test_two_row_observation_training.py)
  ;;
 probe)
  export CUDA_VISIBLE_DEVICES=1
  OUT="$P/runs/observed_qwen_prefix_replay_probe_v1"
  ARGS=(-m scripts.audit_two_row_qwen_prefix_replay --config "$SRC/configs/two_row_qwen_prefix_replay_probe_v1.json" --model "$P/data/qwen3-vl-2b-instruct-89644892" --data "$P/data/observation_two_row_composite108_v1" --head "$P/runs/observed_two_row_composite108_v1/peak_seed0/last.pt" --output "$OUT/probe")
  ;;
 *) exit 2;;
esac
[[ ! -e "$OUT/$STAGE.status.json" ]] || exit 2
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$STAGE" --resume-strategy none -- "$PY" "${ARGS[@]}"
