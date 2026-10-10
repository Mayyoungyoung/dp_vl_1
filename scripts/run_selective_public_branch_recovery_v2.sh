#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
"$P/.venv/bin/python" - "$R" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
assert json.loads((r/'jobs/body_crossing_full_TRAIN_study_v1/receipt.json').read_text())['status']=='completed'
assert json.loads((r/'jobs/public_panda_joint_contract_v1/receipt.json').read_text())['status']=='failed'
assert not (r/'jobs/body_branch_observability_TRAIN_v1').exists()
PY
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]];do sleep 5;done
bash "$S/scripts/launch_selective_repair_v1.sh" --id public_panda_joint_contract_v2 -- bash "$S/scripts/render_selective_body_v1.sh" public_contract --name public_panda_joint_contract_v2
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_branch_observability_TRAIN_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.branch_observability_diagnostic --name body_branch_observability_TRAIN_v1 --dataset "$R/body_events_data_v1/samples.npz"
