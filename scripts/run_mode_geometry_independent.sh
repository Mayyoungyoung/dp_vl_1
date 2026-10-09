#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_ind_$1" -- timeout "$2" "$PY" "${@:3}"; }
job tests 40 -m pytest tests/test_mode_geometry.py tests/test_paired_modes_evaluation.py -q
for arm in C D; do
  job "${arm}_train" 180 -m scripts.mode_geometry_experiment train --arm "$arm" --seed 0 --steps 1200 --name "ind_${arm}_seed0" --independent-decoder
  job "${arm}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "ind_${arm}_seed0" --sampling adaptive
done
job analysis 100 -m scripts.analyze_mode_geometry --names full_B0_seed0 full_C_seed0:adaptive full_D_seed0:adaptive ind_C_seed0:adaptive ind_D_seed0:adaptive --output independent_seed0_analysis
job intervention 100 -m scripts.diagnose_mode_geometry diagnose --names ind_C_seed0 ind_D_seed0 --output independent_train_intervention.json
