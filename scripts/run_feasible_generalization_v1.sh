#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/feasible_space_v1"
D="$P/data/feasible_space_generalization_v1"
bash "$L" --id generalization_prepare_v2 -- "$P/.venv/bin/python" -m research_feasible_space_v1.generalization_data prepare
bash "$L" --id generalization_render_v1 -- bash "$S/scripts/render_feasible_generalization_v1.sh"
bash "$L" --id generalization_export_v1 -- "$P/.venv/bin/python" -m research_feasible_space_v1.generalization_data export
bash "$L" --id generalization_qwen_v1 -- "$P/.venv-qwen/bin/python" -m scripts.observation_cache_qwen --model "$P/data/qwen3-vl-2b-instruct-89644892" --device cuda --threads 4 --gpu-memory-fraction .35 --manifest "$D/export/observations.jsonl" --output "$D/export/qwen_cache"
for seed in 0 1 2; do
  for arm in xyz bounded; do
    bash "$L" --id generalization_${arm}_seed${seed} -- "$P/.venv/bin/python" -m research_feasible_space_v1.evaluate --name generalization_${arm}_seed${seed} --checkpoint "$R/tapered_${arm}_seed${seed}/last.pt" --head "$R/success_tapered_${arm}_seed${seed}/last.pt" --data-root "$D" --expected-requests 336
  done
  for arm in projection center boundcenter; do
    bash "$L" --id generalization_refitted_${arm}_seed${seed} -- "$P/.venv/bin/python" -m research_feasible_space_v1.evaluate --name generalization_refitted_${arm}_seed${seed} --checkpoint "$R/control_${arm}_seed${seed}/last.pt" --head "$R/success_control_${arm}_seed${seed}/last.pt" --data-root "$D" --expected-requests 336
  done
done
bash "$L" --id generalization_statistics_v1 -- "$P/.venv/bin/python" -m research_feasible_space_v1.generalization_stats --name generalization_statistics_v1
