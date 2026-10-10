#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
L="$S/scripts/launch_selective_repair_v1.sh"
while [[ ! -e "$R/body_DEV_execution_suite_seed0_v1/SUMMARY.json" ]];do
 file="$R/jobs/body_DEV_execution_suite_seed0_v1/receipt.json"
 if [[ -e "$file" ]] && grep -q '"status": "failed"' "$file";then echo "Dependency failed:$file";exit 1;fi
 sleep 20
done
while [[ -e "$R/active.lock" ]];do sleep 2;done
bash "$L" --id body_word_safe_tests_v2 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_body_*_v1.py'
for kind in identity lift preserved success planned actual coordinate;do
 name=body_${kind}_DEV_screen_seed0_v2
 bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_screen --name "$name" --kind "$kind" --word-safe --checkpoint "$R/body_recurrent_seed0_v1/last.pt"
done
name=body_actual_unprotected_DEV_screen_seed0_v2
bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_screen --name "$name" --kind actual --word-safe --unprotected --checkpoint "$R/body_recurrent_seed0_v1/last.pt"
name=body_actual_nonrecurrent_DEV_screen_seed0_v2
bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.body_screen --name "$name" --kind actual --word-safe --checkpoint "$R/body_nonrecurrent_seed0_v1/last.pt"
bash "$L" --id body_DEV_execution_suite_seed0_v2 -- bash "$S/scripts/render_selective_body_v1.sh" suite --name body_DEV_execution_suite_seed0_v2 --version v2
