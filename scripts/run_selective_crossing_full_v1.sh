#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ ! -e "$R/jobs/body_binary_seed2_v1/receipt.json" ]] || ! grep -q '"status": "completed"' "$R/jobs/body_binary_seed2_v1/receipt.json";do
 for job in body_feedback_data_v3 body_events_data_v1 body_binary_seed0_v1 body_binary_seed1_v1 body_binary_seed2_v1;do
  file="$R/jobs/$job/receipt.json"
  if [[ -e "$file" ]] && grep -q '"status": "failed"' "$file";then echo "Dependency failed:$file";exit 1;fi
 done
 sleep 20
done
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
exec bash "$S/scripts/run_selective_crossing_study_v1.sh" --name body_crossing_full_TRAIN_study_v1 --full
