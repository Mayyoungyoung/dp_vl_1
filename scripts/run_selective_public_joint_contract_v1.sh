#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ ! -e "$R/jobs/body_crossing_full_TRAIN_study_v1/receipt.json" ]] || ! grep -q '"status": "completed"' "$R/jobs/body_crossing_full_TRAIN_study_v1/receipt.json";do
 if [[ -e "$R/jobs/body_crossing_full_TRAIN_study_v1/receipt.json" ]] && grep -q '"status": "failed"' "$R/jobs/body_crossing_full_TRAIN_study_v1/receipt.json";then exit 1;fi
 sleep 5
done
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id public_panda_joint_contract_v1 -- bash "$S/scripts/render_selective_body_v1.sh" public_contract --name public_panda_joint_contract_v1
