#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
while ! grep -q '"status": "failed"' "$R/jobs/body_full_DEV_returned4_suite_seed0_v1/receipt.json";do sleep 5;done
"$P/.venv/bin/python" - "$R" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);log=(r/'jobs/body_full_DEV_returned4_suite_seed0_v1/stdout.log').read_text()
assert 'TypeError' in log and 'audit()' in log
assert (r/'body_full_identity_DEV_returned4_seed0_v1/RESULTS.json').exists()
assert not (r/'jobs/body_full_DEV_returned4_suite_seed0_v2').exists()
PY
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_full_DEV_returned4_suite_seed0_v2 -- bash "$S/scripts/render_selective_body_v1.sh" full_body_suite --name body_full_DEV_returned4_suite_seed0_v2
