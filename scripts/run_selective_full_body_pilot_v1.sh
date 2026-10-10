#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
"$P/.venv/bin/python" - "$R" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
for n in ('body_crossing_full_TRAIN_study_v1','body_crossing_constant_TRAIN_v1'):
 assert json.loads((r/'jobs'/n/'receipt.json').read_text())['status']=='completed'
PY
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_full_DEV_screens_seed0_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.full_body_pilot --screen
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_full_DEV_returned4_suite_seed0_v1 -- bash "$S/scripts/render_selective_body_v1.sh" full_body_suite --name body_full_DEV_returned4_suite_seed0_v1
