#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
L="$S/scripts/launch_selective_repair_v1.sh"
while [[ ! -e "$R/body_feedback_data_v3/MANIFEST.json" ]];do
 for job in body_feedback_data_v3 body_execution_train_identity_families0_7_targets12_v3 body_execution_train_lift_families0_7_targets12_v3 body_execution_train_preserved_families0_7_targets12_v3 body_execution_train_identity_families8_15_targets012_v3 body_execution_train_lift_families8_15_targets012_v3 body_execution_train_preserved_families8_15_targets012_v3;do
  file="$R/jobs/$job/receipt.json"
  if [[ -e "$file" ]] && grep -q '"status": "failed"' "$file";then echo "Dependency failed:$file";exit 1;fi
 done
 sleep 20
done
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$L" --id body_event_full_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_body_prefix*_v1.py'
bash "$L" --id body_events_data_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_prefix_data --name body_events_data_v1 --dataset "$R/body_feedback_data_v3/samples.npz" --model "$R/public_panda_canonical_v1.npz" --expanded --event-only
for kind in recurrent nonrecurrent;do
 for seed in 0 1 2;do
  name=body_event_${kind}_seed${seed}_v1
  bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_event_forecast --name "$name" --dataset "$R/body_events_data_v1/samples.npz" --kind "$kind" --seed "$seed"
 done
done
for seed in 0 1 2;do
 name=body_binary_seed${seed}_v1
 bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_binary_forecast --name "$name" --dataset "$R/body_feedback_data_v3/samples.npz" --seed "$seed"
done
# End on frozen full TRAIN diagnostics. Actual interpolation,matching heads and
# fresh confirmation remain separate decisions,not inferred from fit completion.
