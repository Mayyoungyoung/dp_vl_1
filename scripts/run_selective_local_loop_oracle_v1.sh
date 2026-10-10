#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
R=/home/wzy/dpvlm/route_set_v1/runs/selective_repair_v1
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_local_loop_support_TRAIN_v1 -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_selective_repair_v1.local_loop_support --name body_local_loop_support_TRAIN_v1
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_local_loop_oracle_TRAIN_v1 -- bash "$S/scripts/render_selective_body_v1.sh" local_loop --name body_local_loop_oracle_TRAIN_v1 --support body_local_loop_support_TRAIN_v1
