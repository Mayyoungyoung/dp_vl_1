#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
for geo in ordinary gap hard; do
  checkpoint="/home/wzy/dpvlm/route_set_v1/runs/realized_coverage_v1/geo_${geo}_seed0/step600.pt"
  launch "rc_refresh_${geo}" -m research_realized_coverage_v1.feedback --name "feedback_geo_${geo}600_train" --split TRAIN --checkpoint "$checkpoint"
  for head in success dense; do
    launch "rc_joint_${geo}_${head}_fit" -m research_realized_coverage_v1.train_allocation --kind "$head" --name "joint_${geo}_${head}" --feedback "feedback_geo_${geo}600_train" --steps 2400
    extra=()
    if [[ "$head" = dense ]]; then extra=(--start-head "joint_${geo}_success/last.pt"); fi
    launch "rc_joint_${geo}_${head}_eval" -m research_realized_coverage_v1.evaluate --name "joint_${geo}_${head}" --head "joint_${geo}_${head}/last.pt" --checkpoint "$checkpoint" "${extra[@]}"
  done
done
