#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_diag2_$1" -- timeout "$2" "$PY" "${@:3}"; }
job tests 40 -m pytest tests/test_mode_geometry.py tests/test_paired_modes_evaluation.py -q
job train_intervention 100 -m scripts.diagnose_mode_geometry diagnose --names full_C_seed0 full_D_seed0 --output train_intervention_v2.json
job resume_full 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --name resume_full
job resume_half 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --stop-after 50 --name resume_split
job resume_finish 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --resume --name resume_split
job resume_compare 40 -m scripts.diagnose_mode_geometry resume --names resume_full resume_split --output resume_exact.json
for arm in C D; do
  job "adaptive_${arm}" 80 -m scripts.mode_geometry_experiment evaluate --name "full_${arm}_seed0" --sampling adaptive
done
