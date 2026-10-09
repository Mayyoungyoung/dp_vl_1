#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_final_$1" -- timeout "$2" "$PY" "${@:3}"; }
for seed in 1 2; do
  job "D2_${seed}_train" 180 -m scripts.mode_geometry_experiment train --arm D --seed "$seed" --steps 1200 --pair-allocation --name "canonical_D2_seed${seed}"
  job "D2_${seed}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_D2_seed${seed}" --sampling adaptive
done
for sampler in ordinary balanced; do
  job "D2_${sampler}" 80 -m scripts.mode_geometry_experiment evaluate --name canonical_D2_seed0 --sampling "$sampler"
done
names=()
for arm in B0 Bset C D D2; do for seed in 0 1 2; do names+=("canonical_${arm}_seed${seed}:adaptive"); done; done
job analysis 120 -m scripts.analyze_mode_geometry --names "${names[@]}" --output final_analysis
job summary 80 -m scripts.summarize_mode_geometry --groups B0=canonical_B0 Bset=canonical_Bset C=canonical_C D=canonical_D D2=canonical_D2 --analysis final_analysis --output final_summary
job figures 100 -m scripts.analyze_mode_geometry --names canonical_Bset_seed0:adaptive canonical_C_seed0:adaptive canonical_D2_seed0:adaptive --output final_visualization --figures
job deployment 60 -m scripts.check_mode_geometry_deployment --name canonical_D2_seed0 --output DEPLOYMENT_REPLAY.json
