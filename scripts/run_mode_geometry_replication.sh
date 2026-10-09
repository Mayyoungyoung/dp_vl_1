#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_rep_$1" -- timeout "$2" "$PY" "${@:3}"; }
for seed in 1 2; do
  job "B0_${seed}_train" 200 -m scripts.mode_geometry_experiment train --arm B --pair-weight 0 --seed "$seed" --steps 1200 --name "canonical_B0_seed${seed}"
  job "B0_${seed}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_B0_seed${seed}" --sampling adaptive
  job "Bset_${seed}_train" 200 -m scripts.mode_geometry_experiment train --arm B --pair-weight 0 --full-positive-set --seed "$seed" --steps 1200 --name "canonical_Bset_seed${seed}"
  job "Bset_${seed}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_Bset_seed${seed}" --sampling adaptive
  for arm in C D; do
    job "${arm}_${seed}_train" 180 -m scripts.mode_geometry_experiment train --arm "$arm" --seed "$seed" --steps 1200 --name "canonical_${arm}_seed${seed}"
    job "${arm}_${seed}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_${arm}_seed${seed}" --sampling adaptive
  done
done
names=()
for arm in B0 Bset C D; do for seed in 0 1 2; do names+=("canonical_${arm}_seed${seed}:adaptive"); done; done
job analysis 120 -m scripts.analyze_mode_geometry --names "${names[@]}" --output three_seed_analysis
job summary 80 -m scripts.summarize_mode_geometry --groups B0=canonical_B0 Bset=canonical_Bset C=canonical_C D=canonical_D --analysis three_seed_analysis --output three_seed_summary
job figures 100 -m scripts.analyze_mode_geometry --names canonical_Bset_seed0:adaptive canonical_C_seed0:adaptive canonical_D_seed0:adaptive --output final_figures --figures
