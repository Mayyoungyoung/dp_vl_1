#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_prefix_conditioning_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_body_prefix_conditioning_v1.py'
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_prefix_conditioning_TRAIN_support_v1 -- "$P/.venv-sim/bin/python" -m research_selective_repair_v1.prefix_conditioning --name body_prefix_conditioning_TRAIN_support_v1
