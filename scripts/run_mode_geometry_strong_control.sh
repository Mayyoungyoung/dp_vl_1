#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_strong_$1" -- timeout "$2" "$PY" "${@:3}"; }
job Bset_train 200 -m scripts.mode_geometry_experiment train --arm B --pair-weight 0 --full-positive-set --seed 0 --steps 1200 --name canonical_Bset_seed0
job Bset_eval 80 -m scripts.mode_geometry_experiment evaluate --name canonical_Bset_seed0 --sampling adaptive
job analysis 100 -m scripts.analyze_mode_geometry --names canonical_Bset_seed0:adaptive canonical_C_seed0:adaptive canonical_D_seed0:adaptive canonical_ind_C_seed0:adaptive canonical_ind_D_seed0:adaptive --output strong_seed0_analysis
for arm in C D; do
  job "${arm}_ordinary" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_${arm}_seed0" --sampling ordinary
  job "${arm}_balanced" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_${arm}_seed0" --sampling balanced
done
