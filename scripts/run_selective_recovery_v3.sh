#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
D="$P/data/selective_repair_interventions_v1"
bash "$L" --id early_executor_B0_v3 -- bash "$S/scripts/render_selective_repair_v1.sh" execute --name early_executor_B0_v3 --pool "$R/baseline_zero_v1/pool.npz" --rows "$R/baseline_zero_v1/rows.json"
bash "$L" --id early_executor_rule_v3 -- bash "$S/scripts/render_selective_repair_v1.sh" execute --name early_executor_rule_v3 --pool "$R/goal_rule_.025_v1/pool.npz" --rows "$R/goal_rule_.025_v1/rows.json"
bash "$L" --id physical_interventions_prepare_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.data prepare
bash "$L" --id physical_interventions_render_v1 -- bash "$S/scripts/render_selective_repair_v1.sh"
bash "$L" --id physical_interventions_export_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.data export
bash "$L" --id physical_interventions_qwen_v1 -- "$P/.venv-qwen/bin/python" -m scripts.observation_cache_qwen --model "$P/data/qwen3-vl-2b-instruct-89644892" --device cuda --threads 4 --gpu-memory-fraction .35 --manifest "$D/export/observations.jsonl" --output "$D/export/qwen_cache"
bash "$L" --id physical_interventions_targets_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.prepare --name interventions_prepared_v1 --data "$D"
bash "$L" --id physical_interventions_goal_targets_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.goal_data --name interventions_goal_v1 --source interventions_prepared_v1 --data "$D" --prototype "$R/goal_prepared_v1/PROTOTYPES.json"
