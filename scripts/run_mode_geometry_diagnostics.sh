#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_diag_$1" -- timeout "$2" "$PY" "${@:3}"; }
job train_intervention 100 -m scripts.diagnose_mode_geometry diagnose --names full_C_seed0 full_D_seed0 --output train_intervention.json
job resume_full 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --name resume_full
job resume_half 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --stop-after 50 --name resume_split
job resume_finish 80 -m scripts.mode_geometry_experiment train --arm D --seed 91 --steps 100 --resume --name resume_split
job resume_compare 40 -m scripts.diagnose_mode_geometry resume --names resume_full resume_split --output resume_exact.json
