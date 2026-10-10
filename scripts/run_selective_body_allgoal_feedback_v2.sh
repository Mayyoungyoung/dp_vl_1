#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
L="$S/scripts/launch_selective_repair_v1.sh"
while [[ ! -e "$R/body_DEV_execution_suite_seed0_v2/SUMMARY.json" ]];do
 for job in body_word_safe_tests_v2 body_DEV_execution_suite_seed0_v1 body_DEV_execution_suite_seed0_v2;do
  file="$R/jobs/$job/receipt.json"
  if [[ -e "$file" ]] && grep -q '"status": "failed"' "$file";then echo "Dependency failed:$file";exit 1;fi
 done
 sleep 20
done
while [[ -e "$R/active.lock" ]];do sleep 2;done
for scope in families0_7_targets12 families8_15_targets012;do
 start=0;targets=1,2
 [[ "$scope" == families8_15_targets012 ]] && start=8 && targets=0,1,2
 for option in identity lift preserved;do
  name=body_execution_train_${option}_${scope}_v2
  args=(--lift 0)
  [[ "$option" != identity ]] && args=(--lift .08)
  [[ "$option" == preserved ]] && args+=(--preserve-row-crossings)
  bash "$L" --id "$name" -- bash "$S/scripts/render_selective_body_v1.sh" execution --name "$name" --pool "$R/constraints_global_interventions_TRAIN_body_v1/pool.npz" --family-start "$start" --families 8 --targets "$targets" "${args[@]}"
  bash "$L" --id ${name}_semantics_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name ${name}_semantics_v2 --source "$name"
 done
done
bash "$L" --id body_feedback_data_v2 -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_feedback_data --name body_feedback_data_v2 --expanded
