#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
for arm in ordinary kl_only gap hard; do
  extra=()
  if [[ "$arm" = gap || "$arm" = hard ]]; then extra=(--head net_fit0/step2400.pt); fi
  launch "rc_geo_${arm}_train0" -m research_realized_coverage_v1.train_geometry --arm "$arm" --name "geo_${arm}_seed0" "${extra[@]}"
  for step in 600 1800 3600; do
    launch "rc_geo_${arm}_eval${step}" -m research_realized_coverage_v1.evaluate --name "geo_${arm}_${step}_adaptive" --checkpoint "/home/wzy/dpvlm/route_set_v1/runs/realized_coverage_v1/geo_${arm}_seed0/step${step}.pt"
  done
done
