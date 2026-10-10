#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
L="$S/scripts/launch_selective_repair_v1.sh"
# Resume after the verified completed immutable test job. Never reuse job IDs.
"$P/.venv/bin/python" - "$R" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);d=json.loads((r/'jobs/body_event_full_tests_v1/receipt.json').read_text())
assert d['status']=='completed' and d['exit_code']==0
assert d['source_commit']=='3271e227da4fe0cdf58ae827ae23ea1482d6b437'
assert (r/'body_feedback_data_v3/MANIFEST.json').exists()
for n in ['body_events_data_v1']+[f'body_event_{k}_seed{s}_v1' for k in ('recurrent','nonrecurrent') for s in range(3)]+[f'body_binary_seed{s}_v1' for s in range(3)]:
 assert not (r/'jobs'/n).exists(), n
PY
while [[ -e "$R/active.lock" || -e "$R/prefix_pilot_gpu.lock" || -e "$R/body_aux_cpu.lock" ]]; do sleep 5; done
bash "$L" --id body_events_data_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_prefix_data --name body_events_data_v1 --dataset "$R/body_feedback_data_v3/samples.npz" --model "$R/public_panda_canonical_v1.npz" --expanded --event-only
for kind in recurrent nonrecurrent; do
 for seed in 0 1 2; do
  name=body_event_${kind}_seed${seed}_v1
  bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_event_forecast --name "$name" --dataset "$R/body_events_data_v1/samples.npz" --kind "$kind" --seed "$seed"
 done
done
for seed in 0 1 2; do
 name=body_binary_seed${seed}_v1
 bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_binary_forecast --name "$name" --dataset "$R/body_feedback_data_v3/samples.npz" --seed "$seed"
done
