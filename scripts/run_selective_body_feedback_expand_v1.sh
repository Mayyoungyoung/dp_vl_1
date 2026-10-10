#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
for option in identity lift preserved;do
 name=body_execution_train_${option}_families2_7_v1
 args=(--lift 0)
 [[ "$option" != identity ]] && args=(--lift .08)
 [[ "$option" == preserved ]] && args+=(--preserve-row-crossings)
 bash "$L" --id "$name" -- bash "$S/scripts/render_selective_body_v1.sh" execution --name "$name" --pool "$R/constraints_global_interventions_TRAIN_body_v1/pool.npz" --family-start 2 --families 6 --targets 0 "${args[@]}"
 bash "$L" --id ${name}_semantics_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.execution_semantics --name ${name}_semantics_v1 --source "$name"
done
