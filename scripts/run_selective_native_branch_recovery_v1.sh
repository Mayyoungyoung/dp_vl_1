#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
R=/home/wzy/dpvlm/route_set_v1/runs/selective_repair_v1
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
[[ -f "$R/body_native_branch_data_v1/MANIFEST.json" ]]
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_native_branch_study_seed0_v2 -- "$P" -m research_selective_repair_v1.native_branch_study --name body_native_branch_study_seed0_v2
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_native_branch_API_pilot_v1 -- "$P" -m research_selective_repair_v1.native_branch_api --name body_native_branch_API_pilot_v1
