#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
job() { bash scripts/launch_verified_set_v1.sh --id "mg_alloc_$1" -- timeout "$2" "$PY" "${@:3}"; }
job tests 40 -m pytest tests/test_mode_geometry.py tests/test_paired_modes_evaluation.py -q
job D2_train 180 -m scripts.mode_geometry_experiment train --arm D --seed 0 --steps 1200 --pair-allocation --name canonical_D2_seed0
job D2_eval 80 -m scripts.mode_geometry_experiment evaluate --name canonical_D2_seed0 --sampling adaptive
job analysis 100 -m scripts.analyze_mode_geometry --names canonical_Bset_seed0:adaptive canonical_C_seed0:adaptive canonical_D_seed0:adaptive canonical_D2_seed0:adaptive --output allocation_fix_seed0_analysis
