#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ ! -e "$R/jobs/body_signed_events_pilot_v1/receipt.json" ]] || ! grep -q '"status": "completed"' "$R/jobs/body_signed_events_pilot_v1/receipt.json";do
 for n in body_execution_train_signed_lower_pilot_v1 body_signed_events_pilot_v1;do
  if [[ -e "$R/jobs/$n/receipt.json" ]] && grep -q '"status": "failed"' "$R/jobs/$n/receipt.json";then exit 1;fi
 done
 sleep 5
done
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_signed_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_body_prefix*_v1.py'
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_signed_TRAIN_study_seed0_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.signed_study --name body_signed_TRAIN_study_seed0_v1
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_signed_TRAIN_screens_seed0_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.signed_study --name body_signed_TRAIN_screens_seed0_v1 --screen
