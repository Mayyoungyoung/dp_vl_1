#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=8356b09b8436d00a4f98a76ffdd8c7d84033e6ca
SRC="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_route_sft_autoregressive_v1"
test ! -e "$JOBS"
mkdir "$JOBS"
cp "${BASH_SOURCE[0]}" "$JOBS/launcher.sh"
sha256sum "$JOBS/launcher.sh" > "$JOBS/launcher.sha256"
cd "$SRC"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=-1 "$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id fixed_source_tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_vlm_sft_evaluation.py -q
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id best_seed0 --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.evaluate_vlm_route_sft --checkpoint "$P/runs/vlm_route_sft_v1/seed0/best.pt" --output "$JOBS/best_seed0" --seed 0 --repeats 1
export CUDA_VISIBLE_DEVICES=-1
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id analysis_best_seed0 --resume-strategy none -- "$P/.venv/bin/python" -m scripts.analyze_vlm_route_sft --generation "$JOBS/best_seed0" --output "$P/runs/vlm_route_sft_analysis_v1/best_seed0"
