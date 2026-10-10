#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ ! -e "$R/body_actual_nonrecurrent_DEV_screen_seed0_v1/SUMMARY.json" ]];do
 for job in body_forecast_tests_v1 body_feedback_data_v1 body_recurrent_seed0_v1 body_recurrent_seed1_v1 body_recurrent_seed2_v1 body_nonrecurrent_seed0_v1 body_nonrecurrent_seed1_v1 body_nonrecurrent_seed2_v1 body_actual_nonrecurrent_DEV_screen_seed0_v1;do
  file="$R/jobs/$job/receipt.json"
  if [[ -e "$file" ]] && grep -q '"status": "failed"' "$file";then echo "Dependency failed:$file";exit 1;fi
 done
 sleep 20
done
while [[ -e "$R/active.lock" ]];do sleep 2;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_DEV_execution_suite_seed0_v1 -- bash "$S/scripts/render_selective_body_v1.sh" suite --name body_DEV_execution_suite_seed0_v1
