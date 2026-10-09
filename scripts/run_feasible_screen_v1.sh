#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
for arm in xyz bounded relative; do
  bash "$L" --id screen_${arm}_seed0 -- "$P" -m research_feasible_space_v1.train --name screen_${arm}_seed0 --arm "$arm" --steps 2400
  bash "$L" --id eval_screen_${arm}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name screen_${arm}_seed0 --checkpoint "$R/screen_${arm}_seed0/last.pt" --perturb
done
bash "$L" --id screen_peer_seed0 -- "$P" -m research_feasible_space_v1.train --name screen_peer_seed0 --arm bounded --peer-boundaries --steps 2400
bash "$L" --id eval_screen_peer_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name screen_peer_seed0 --checkpoint "$R/screen_peer_seed0/last.pt" --perturb
for kind in center projection; do
  bash "$L" --id eval_${kind}_xyz_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name ${kind}_xyz_seed0 --checkpoint "$R/screen_xyz_seed0/last.pt" --kind "$kind"
done
bash "$L" --id eval_center_bounded_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name center_bounded_seed0 --checkpoint "$R/screen_bounded_seed0/last.pt" --kind center
