#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ ! -e "$R/jobs/public_panda_joint_contract_v1/receipt.json" ]] || ! grep -q '"status": "completed"' "$R/jobs/public_panda_joint_contract_v1/receipt.json";do
 if [[ -e "$R/jobs/public_panda_joint_contract_v1/receipt.json" ]] && grep -q '"status": "failed"' "$R/jobs/public_panda_joint_contract_v1/receipt.json";then exit 1;fi
 sleep 5
done
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_branch_observability_TRAIN_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.branch_observability_diagnostic --name body_branch_observability_TRAIN_v1 --dataset "$R/body_events_data_v1/samples.npz"
