#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
D="$P/data/selective_repair_interventions_v1"
for kind in global protected mode_global;do
 name=constraints_${kind}_interventions_v1
 bash "$L" --id ${name}_rows -- "$P/.venv/bin/python" -m research_selective_repair_v1.sealed_rows --name ${name}_rows --pool "$R/$name/pool.npz" --data "$D"
 bash "$L" --id ${name}_transport -- "$P/.venv/bin/python" -m research_selective_repair_v1.intervention_audit --name ${name}_transport --pool "$R/$name/pool.npz" --rows "$R/${name}_rows/rows.json" --data "$D"
done
for kind in zero geometry;do
 name=interventions_${kind}_calibrated_v1
 bash "$L" --id ${name}_transport -- "$P/.venv/bin/python" -m research_selective_repair_v1.intervention_audit --name ${name}_transport --pool "$R/$name/pool.npz" --rows "$R/$name/rows.json" --data "$D"
done
# Same fixed two DEV families/open+closed/target0, actual returned4, no retries.
# Diagnostics of body bottlenecks; no accepted paper advantage from16routes.
for kind in global protected mode_global;do
 name=constraints_${kind}_interventions_v1
 bash "$L" --id ${name}_executor -- env LC_ALL=C LANG=C PYTHONUTF8=1 "$P/.venv-sim/bin/python" -m research_selective_repair_v1.execute --name ${name}_executor --pool "$R/$name/pool.npz" --rows "$R/${name}_rows/rows.json" --data "$D"
done
