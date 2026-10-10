#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
R=/home/wzy/dpvlm/route_set_v1/runs/selective_repair_v1
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_prefix_native_TRAIN_fit_support_v1 -- bash "$S/scripts/render_selective_body_v1.sh" prefix_probe --name body_prefix_native_TRAIN_fit_support_v1 --family-indices 0 1 2 3
