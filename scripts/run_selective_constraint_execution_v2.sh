#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
D="$P/data/selective_repair_interventions_v1"
for kind in global protected mode_global;do
 name=constraints_${kind}_interventions_v1
 bash "$L" --id ${name}_transport_common -- "$P/.venv/bin/python" -m research_selective_repair_v1.intervention_audit --name ${name}_transport_common --pool "$R/$name/pool.npz" --rows "$R/${name}_rows/rows.json" --data "$D" --reference-pool "$R/interventions_zero_calibrated_v1/pool.npz" --reference-rows "$R/interventions_zero_calibrated_v1/rows.json"
done
for kind in zero geometry;do
 name=interventions_${kind}_calibrated_v1
 bash "$L" --id ${name}_transport_common -- "$P/.venv/bin/python" -m research_selective_repair_v1.intervention_audit --name ${name}_transport_common --pool "$R/$name/pool.npz" --rows "$R/$name/rows.json" --data "$D" --reference-pool "$R/interventions_zero_calibrated_v1/pool.npz" --reference-rows "$R/interventions_zero_calibrated_v1/rows.json"
done
for kind in global protected mode_global;do
 name=constraints_${kind}_interventions_v1
 bash "$L" --id ${name}_executor_v2 -- bash "$S/scripts/render_selective_repair_v1.sh" execute --name ${name}_executor_v2 --pool "$R/$name/pool.npz" --rows "$R/${name}_rows/rows.json" --data "$D"
done
