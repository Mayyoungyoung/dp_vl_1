#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
# Fixed TRAIN-estimated first-row systematic overshoot, not a scale sweep.
AMP=$(cd "$S" && "$P/.venv-sim/bin/python" -c 'from research_selective_repair_v1.signed_feedback import lowering_amplitude; print(lowering_amplitude())')
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_execution_train_signed_lower_pilot_v1 -- bash "$S/scripts/render_selective_body_v1.sh" execution --name body_execution_train_signed_lower_pilot_v1 --pool "$R/constraints_global_interventions_TRAIN_body_v1/pool.npz" --family-indices 0,1,2,3,14,15 --targets 0 --lift "$AMP"
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_execution_train_signed_lower_pilot_v1_semantics_v1 -- "$P/.venv-sim/bin/python" -m research_selective_repair_v1.execution_semantics --name body_execution_train_signed_lower_pilot_v1_semantics_v1 --source body_execution_train_signed_lower_pilot_v1
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_signed_events_pilot_v1 -- "$P/.venv-sim/bin/python" -m research_selective_repair_v1.signed_feedback --name body_signed_events_pilot_v1
