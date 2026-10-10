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
while [[ -e "$R/active.lock" ]];do sleep 2;done
bash "$L" --id body_prefix_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_body_*_v1.py'
bash "$L" --id public_kinematics_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_public_kinematics_v1.py'
bash "$L" --id body_prefix_data_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_prefix_data --name body_prefix_data_v1 --dataset "$R/body_feedback_data_v3/samples.npz" --model "$R/public_panda_canonical_v1.npz" --expanded
bash "$L" --id body_prefix_recovery_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.check_prefix_recovery --name body_prefix_recovery_v1 --dataset "$R/body_prefix_data_v1/samples.npz"
for kind in recurrent nonrecurrent;do
 for seed in 0 1 2;do
  name=body_prefix_${kind}_seed${seed}_v1
  bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_prefix_forecast --name "$name" --dataset "$R/body_prefix_data_v1/samples.npz" --kind "$kind" --seed "$seed"
 done
done
for seed in 0 1 2;do
 name=body_binary_seed${seed}_v1
 bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_binary_forecast --name "$name" --dataset "$R/body_feedback_data_v3/samples.npz" --seed "$seed"
done
# Deliberately end on TRAIN diagnostics. No automatic DEV evaluation before
# checking whether free prefix-state transport is meaningful on held families.
