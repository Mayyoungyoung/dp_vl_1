#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
[[ -e "$R/body_execution_train_lift.08_pilot_v1/SUMMARY.json" ]] || { echo 'Full teacher incomplete';exit 1; }
bash "$L" --id execution_semantics_tests_v2 -- "$P/.venv/bin/python" -m unittest discover -s tests -p test_execution_semantics_v1.py
for kind in global protected mode_global;do
 name=constraints_${kind}_interventions_v1_executor_v2
 bash "$L" --id ${name}_semantics_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name ${name}_semantics_v2 --source "$name"
done
for lift in 0 .08;do
 name=body_execution_train_lift${lift}_pilot_v1
 bash "$L" --id ${name}_semantics_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name ${name}_semantics_v2 --source "$name"
done
bash "$L" --id body_execution_train_preservedlift_pilot_v1 -- bash "$S/scripts/render_selective_body_v1.sh" execution --name body_execution_train_preservedlift_pilot_v1 --pool "$R/constraints_global_interventions_TRAIN_body_v1/pool.npz" --families 2 --lift .08 --preserve-row-crossings
bash "$L" --id body_execution_train_preservedlift_pilot_v1_semantics_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name body_execution_train_preservedlift_pilot_v1_semantics_v2 --source body_execution_train_preservedlift_pilot_v1
