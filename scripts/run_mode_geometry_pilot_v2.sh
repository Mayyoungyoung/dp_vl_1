#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_v2_$1" -- timeout "$2" "$PY" "${@:3}"; }
job tests 40 -m pytest tests/test_mode_geometry.py tests/test_paired_modes_evaluation.py -q
job prepare 180 -m scripts.mode_geometry_experiment prepare
for arm in B C D; do
  job "pilot_${arm}_train" 220 -m scripts.mode_geometry_experiment train --arm "$arm" --seed 0 --steps 400 --name "pilot_${arm}_seed0"
  job "pilot_${arm}_eval" 120 -m scripts.mode_geometry_experiment evaluate --name "pilot_${arm}_seed0"
done
