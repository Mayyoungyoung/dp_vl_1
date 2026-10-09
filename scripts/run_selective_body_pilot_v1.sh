#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
# Wait for the already authorized fixed executor coordinator's own final receipt.
while [[ ! -e "$P/runs/selective_repair_v1/constraints_mode_global_interventions_v1_executor_v2/RESULTS.json" ]];do
 if [[ -e "$P/runs/selective_repair_v1/jobs/constraints_mode_global_interventions_v1_executor_v2/receipt.json" ]];then
  status=$("$P/.venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$P/runs/selective_repair_v1/jobs/constraints_mode_global_interventions_v1_executor_v2/receipt.json")
  [[ "$status" != failed ]] || { echo 'Executor technical failure; inspect before proceeding';exit 1; }
 fi
 sleep 5
done
bash "$S/scripts/launch_selective_repair_v1.sh" --id body_ik_train_pilot_v1 -- bash "$S/scripts/render_selective_body_v1.sh" --name body_ik_train_pilot_v1 --families 2
