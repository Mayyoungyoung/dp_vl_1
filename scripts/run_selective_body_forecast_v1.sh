#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
# Wait for the already running immutable feedback coordinator; never duplicate it.
while [[ ! -e "$R/body_execution_train_preserved_families2_7_v1_semantics_v1/SUMMARY.json" ]];do
 for option in identity lift preserved;do
  file="$R/jobs/body_execution_train_${option}_families2_7_v1/receipt.json"
  if [[ -e "$file" ]] && grep -q '"status": "failed"' "$file";then echo "Feedback failed:$file";exit 1;fi
 done
 sleep 20
done
while [[ -e "$R/active.lock" ]];do sleep 2;done
bash "$L" --id body_forecast_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_body_*_v1.py'
bash "$L" --id body_feedback_data_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_feedback_data --name body_feedback_data_v1
for kind in recurrent nonrecurrent;do
 for seed in 0 1 2;do
  name=body_${kind}_seed${seed}_v1
  bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_forecast --name "$name" --dataset "$R/body_feedback_data_v1/samples.npz" --kind "$kind" --seed "$seed"
 done
done
for kind in identity lift preserved success planned actual coordinate;do
 name=body_${kind}_DEV_screen_seed0_v1
 bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_screen --name "$name" --kind "$kind" --checkpoint "$R/body_recurrent_seed0_v1/last.pt"
done
name=body_actual_unprotected_DEV_screen_seed0_v1
bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_screen --name "$name" --kind actual --unprotected --checkpoint "$R/body_recurrent_seed0_v1/last.pt"
name=body_actual_nonrecurrent_DEV_screen_seed0_v1
bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_screen --name "$name" --kind actual --checkpoint "$R/body_nonrecurrent_seed0_v1/last.pt"
