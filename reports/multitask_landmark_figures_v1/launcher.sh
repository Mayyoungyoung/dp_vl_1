#!/usr/bin/env bash
set -euo pipefail
root=/home/wzy/dpvlm/route_set_v1
release=$root/research_v2/releases/e1221123b0ddde4fc70ac3d731568c40e636d0c7
export CODE_COMMIT=e1221123b0ddde4fc70ac3d731568c40e636d0c7
export PYTHONPATH=$release
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
cd "$release"
run=$root/runs/multitask_landmark_figures_v1
test ! -e "$run"
mkdir "$run"
cp "${BASH_SOURCE[0]}" "$run/launcher.sh"
sha256sum scripts/plot_multitask_predictions.py "${BASH_SOURCE[0]}" > "$run/source_sha256.txt"
ordinary=$root/runs/observed_multitask_prefix108_v1/free_offset_seed0
auxiliary=$root/runs/observed_multitask_landmark_aux_v1/event_supported_seed0
snapshot=$root/data/observation_multitask_prefix108_v1
"$root/.venv/bin/python" -m scripts.record_job --output "$run" --run-id ordinary --resume-strategy fresh-output -- "$root/.venv/bin/python" -m scripts.plot_multitask_predictions --run "$ordinary" --comparison-run "$auxiliary" --snapshot "$snapshot" --all-dev-parents --output "$run/ordinary"
"$root/.venv/bin/python" -m scripts.record_job --output "$run" --run-id auxiliary --resume-strategy fresh-output -- "$root/.venv/bin/python" -m scripts.plot_multitask_predictions --run "$auxiliary" --comparison-run "$ordinary" --snapshot "$snapshot" --all-dev-parents --output "$run/auxiliary"
