#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
for seed in 0 1 2; do
  for control in center projection boundcenter; do
    arm=xyz;kind="$control"
    if [[ "$control" == boundcenter ]]; then arm=bounded;kind=center;fi
    n=control_${control}_seed${seed}
    bash "$L" --id view_$n -- "$P" -m research_feasible_space_v1.make_views --name "$n" --parent "$R/tapered_${arm}_seed${seed}/last.pt" --kind "$kind"
    bash "$L" --id feedback_$n -- "$P" -m research_feasible_space_v1.feedback --name feedback_$n --checkpoint "$R/$n/last.pt"
    bash "$L" --id success_$n -- "$P" -m research_feasible_space_v1.feedback --name success_$n --pool feedback_$n --seed "$seed"
    bash "$L" --id eval_refitted_$n -- "$P" -m research_feasible_space_v1.evaluate --name refitted_$n --checkpoint "$R/$n/last.pt" --head "$R/success_$n/last.pt"
  done
done
