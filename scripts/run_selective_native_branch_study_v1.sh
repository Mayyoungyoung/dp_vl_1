#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
R=/home/wzy/dpvlm/route_set_v1/runs/selective_repair_v1
while [[ ! -f "$R/body_prefix_native_TRAIN_fit_support_v1/SUMMARY.json" || -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_native_branch_data_v1 -- "$P" -m research_selective_repair_v1.native_branch_data --name body_native_branch_data_v1
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_native_branch_study_seed0_v1 -- "$P" -m research_selective_repair_v1.native_branch_study --name body_native_branch_study_seed0_v1
