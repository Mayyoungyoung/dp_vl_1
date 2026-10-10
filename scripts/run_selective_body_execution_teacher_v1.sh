#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
while [[ ! -e "$R/body_planning_train_pilot_v1/PLANNING_SUMMARY.json" ]];do
 status=$("$P/.venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$R/jobs/body_planning_train_pilot_v1/receipt.json")
 [[ "$status" != failed ]] || { echo 'Predecessor failed; inspect before continuation';exit 1; }
 sleep 5
done
bash "$L" --id constraints_global_interventions_TRAIN_body_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.constraint_screen --name constraints_global_interventions_TRAIN_body_v1 --checkpoint "$R/constraints_seed0_v1/last.pt" --data-name interventions_calibrated_targets_v1 --data "$P/data/selective_repair_interventions_v1" --kind global --role TRAIN
for lift in 0 .08;do
 name=body_execution_train_lift${lift}_pilot_v1
 bash "$L" --id "$name" -- bash "$S/scripts/render_selective_body_v1.sh" execution --name "$name" --pool "$R/constraints_global_interventions_TRAIN_body_v1/pool.npz" --families 2 --lift "$lift"
done
