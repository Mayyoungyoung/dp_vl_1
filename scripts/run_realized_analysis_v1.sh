#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
launch rc_reference_words -m research_realized_coverage_v1.reference_words
launch rc_deploy_success -m research_realized_coverage_v1.check_deployment --name success_context_step2400 --head success_context_fit0/step2400.pt
launch rc_deploy_dense -m research_realized_coverage_v1.check_deployment --name dense_context_step2400 --head dense_context_fit0/step2400.pt --starter success_context_fit0/step2400.pt
names=(../mode_geometry_v1/canonical_C_seed0:adaptive success_context_step2400:adaptive dense_context_step2400:adaptive net_context_step2400:adaptive no_peer_context_step2400:adaptive added_only_context_step2400:adaptive)
for seed in 1 2 3 4; do
  names+=("success_rep_seed${seed}:adaptive" "dense_rep_seed${seed}:adaptive")
done
for geo in ordinary gap hard; do
  names+=("joint_${geo}_success:adaptive" "joint_${geo}_dense:adaptive")
done
launch rc_fixed_witness_analysis -m research_realized_coverage_v1.analyze --names "${names[@]}" --output fixed_witness_analysis
launch rc_actual_figures -m research_realized_coverage_v1.analyze --names ../mode_geometry_v1/canonical_C_seed0:adaptive success_context_step2400:adaptive dense_context_step2400:adaptive --labels C Success Dense --output actual_figures --figures
