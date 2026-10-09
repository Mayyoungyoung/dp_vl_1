#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
bash "$L" --id constraint_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_selective_constraints_v1.py'
bash "$L" --id constraints_seed0_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.constraints --name constraints_seed0_v1
for population in old interventions;do
 if [[ "$population" == old ]];then data="$P/data/paired_modes_v1_v2";else data="$P/data/selective_repair_interventions_v1";fi
 for kind in global protected mode_global;do
  name=constraints_${kind}_${population}_v1
  bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.constraint_screen --name "$name" --checkpoint "$R/constraints_seed0_v1/last.pt" --data-name ${population}_calibrated_targets_v1 --data "$data" --kind "$kind"
 done
done
