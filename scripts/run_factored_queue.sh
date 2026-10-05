#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/home/wzy/dpvlm/route_set_v1
run="$root/runs/factored_q_v1"
py="$root/.venv/bin/python"
mkdir -p "$run/coordinator"
test ! -e "$run/coordinator/started"
date -u +%FT%TZ > "$run/coordinator/started"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/coordinator/closure.json"' EXIT
{ nvidia-smi; uptime; free -h; df -h "$root"; ps -u wzy -o pid,pcpu,pmem,args --sort=-pcpu | head -20 || true; } > "$run/coordinator/resources_before.txt"
for other in "$root/runs/paired_modes_v1/active.lock" "$root/runs/observed_probability_v1/active.lock"; do
  test ! -e "$other"
done
bash scripts/launch_factored_q.sh --id tests -- "$py" -m pytest -q tests/test_factored_q.py
bash scripts/launch_factored_q.sh --id profile_continuous -- "$py" -m scripts.run_factored_q fit --name profile_continuous --steps 100
bash scripts/launch_factored_q.sh --id profile_part1 -- "$py" -m scripts.run_factored_q fit --name profile_resumed --steps 100 --stop-after 50
bash scripts/launch_factored_q.sh --id profile_part2 -- "$py" -m scripts.run_factored_q fit --name profile_resumed --steps 100 --resume
bash scripts/launch_factored_q.sh --id resume_check -- "$py" -m scripts.check_factored_resume
for gs in 0 1 2; do
  for ss in 0 1 2; do
    for arm in single joint marginal conditional; do
      bash scripts/launch_factored_q.sh --id "${arm}_g${gs}_s${ss}" -- "$py" -m scripts.run_factored_q run --arm "$arm" --generator-seed "$gs" --scorer-seed "$ss"
    done
  done
done
bash scripts/launch_factored_q.sh --id package -- "$py" -m scripts.run_factored_q package
bash scripts/launch_factored_q.sh --id public_cli -- "$py" -m scripts.check_factored_cli
bash scripts/launch_factored_q.sh --id analyze -- "$py" -m scripts.analyze_factored_q
