#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_full_$1" -- timeout "$2" "$PY" "${@:3}"; }
job pilot_analysis 100 -m scripts.analyze_mode_geometry --names pilot_B_seed0 pilot_C_seed0 pilot_D_seed0 --output pilot_analysis
job B0_train 180 -m scripts.mode_geometry_experiment train --arm B --pair-weight 0 --seed 0 --steps 1200 --name full_B0_seed0
job B0_eval 80 -m scripts.mode_geometry_experiment evaluate --name full_B0_seed0
for arm in B C D; do
  job "${arm}_train" 180 -m scripts.mode_geometry_experiment train --arm "$arm" --seed 0 --steps 1200 --name "full_${arm}_seed0"
  job "${arm}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "full_${arm}_seed0"
done
job analysis 100 -m scripts.analyze_mode_geometry --names full_B0_seed0 full_B_seed0 full_C_seed0 full_D_seed0 --output full_seed0_analysis --figures
job sampling 80 -m scripts.mode_geometry_experiment evaluate --name full_D_seed0 --sampling ordinary
