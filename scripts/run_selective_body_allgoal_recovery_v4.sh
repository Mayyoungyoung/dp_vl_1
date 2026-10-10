#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
L="$S/scripts/launch_selective_repair_v1.sh"
identity=body_execution_train_identity_families8_15_targets012_v3
[[ -e "$R/$identity/SUMMARY.json" ]]
grep -q '"status": "completed"' "$R/jobs/$identity/receipt.json"
# Recovery begins strictly after this completed384trial teacher. Every next
# output/job name was verified absent; never repeat the completed teacher.
[[ ! -e "$R/${identity}_semantics_v2" && ! -e "$R/jobs/${identity}_semantics_v2" ]]
while [[ -e "$R/active.lock" ]];do sleep 2;done
bash "$L" --id ${identity}_semantics_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name ${identity}_semantics_v2 --source "$identity"
for option in lift preserved;do
 name=body_execution_train_${option}_families8_15_targets012_v3
 [[ ! -e "$R/$name" && ! -e "$R/jobs/$name" ]]
 args=(--lift .08)
 [[ "$option" == preserved ]] && args+=(--preserve-row-crossings)
 bash "$L" --id "$name" -- bash "$S/scripts/render_selective_body_v1.sh" execution --name "$name" --pool "$R/constraints_global_interventions_TRAIN_body_v1/pool.npz" --family-start 8 --families 8 --targets 0,1,2 "${args[@]}"
 bash "$L" --id ${name}_semantics_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name ${name}_semantics_v2 --source "$name"
done
bash "$L" --id body_feedback_data_v3 -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_feedback_data --name body_feedback_data_v3 --expanded
