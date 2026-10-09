#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_can_$1" -- timeout "$2" "$PY" "${@:3}"; }
job tests 40 -m pytest tests/test_mode_geometry.py tests/test_paired_modes_evaluation.py -q
job resume_full 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --name canonical_resume_full
job resume_half 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --stop-after 50 --name canonical_resume_split
job resume_finish 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --resume --name canonical_resume_split
job resume_compare 40 -m scripts.diagnose_mode_geometry resume --names canonical_resume_full canonical_resume_split --output canonical_resume_exact.json
job B0_train 180 -m scripts.mode_geometry_experiment train --arm B --pair-weight 0 --seed 0 --steps 1200 --name canonical_B0_seed0
job B0_eval 80 -m scripts.mode_geometry_experiment evaluate --name canonical_B0_seed0 --sampling adaptive
for arm in C D; do
  job "${arm}_train" 180 -m scripts.mode_geometry_experiment train --arm "$arm" --seed 0 --steps 1200 --name "canonical_${arm}_seed0"
  job "${arm}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_${arm}_seed0" --sampling adaptive
  job "ind_${arm}_train" 180 -m scripts.mode_geometry_experiment train --arm "$arm" --seed 0 --steps 1200 --name "canonical_ind_${arm}_seed0" --independent-decoder
  job "ind_${arm}_eval" 80 -m scripts.mode_geometry_experiment evaluate --name "canonical_ind_${arm}_seed0" --sampling adaptive
done
job analysis 100 -m scripts.analyze_mode_geometry --names canonical_B0_seed0:adaptive canonical_C_seed0:adaptive canonical_D_seed0:adaptive canonical_ind_C_seed0:adaptive canonical_ind_D_seed0:adaptive --output canonical_seed0_analysis
job intervention 100 -m scripts.diagnose_mode_geometry diagnose --names canonical_C_seed0 canonical_D_seed0 canonical_ind_C_seed0 canonical_ind_D_seed0 --output canonical_train_intervention.json
