#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
SRC=$P/research_v2/releases/9e0094aff152d48cd34eadd24e554a05f0cf4a0f
cd "$SRC"
export CODE_COMMIT=9e0094aff152d48cd34eadd24e554a05f0cf4a0f PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$P/.venv/bin/python" -m scripts.record_job --output "$P/runs/observed_two_row_prefix76_online_v1" --run-id best --resume-strategy fresh-output -- "$P/.venv-qwen/bin/python" -m scripts.evaluate_observed_two_row_online --run "$P/runs/observed_two_row_prefix76_v1/peak_seed0" --data "$P/data/observation_two_row_prefix76_v1" --model "$P/data/qwen3-vl-2b-instruct-89644892" --training-source "$SRC" --output "$P/runs/observed_two_row_prefix76_online_v1/best"
