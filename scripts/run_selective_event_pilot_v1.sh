#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
R=/home/wzy/dpvlm/route_set_v1/runs/selective_repair_v1
# Serialize after our conditioned state diagnostic; do not touch its export or
# lock. The launched pilot verifies the remaining CPU teacher and GPU limits.
while [[ -e "$R/prefix_pilot_gpu.lock" ]];do sleep 5;done
if [[ ! -e "$R/technical/body_event_cpu_tests_v1/receipt.json" ]] || ! grep -q '"status": "completed"' "$R/technical/body_event_cpu_tests_v1/receipt.json";then exit 1;fi
bash "$S/scripts/run_selective_prefix_pilot_v1.sh" body_event_dense_pilot_v1 --version v1 --object event
