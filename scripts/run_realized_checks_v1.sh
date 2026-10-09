#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
launch rc_resume_head_full -m research_realized_coverage_v1.train_allocation --kind dense --name resume_head_full --steps 100
launch rc_resume_head_half -m research_realized_coverage_v1.train_allocation --kind dense --name resume_head_split --steps 100 --stop-after 50
launch rc_resume_head_finish -m research_realized_coverage_v1.train_allocation --kind dense --name resume_head_split --steps 100 --resume
launch rc_resume_head_compare -m research_realized_coverage_v1.resume_audit --names resume_head_full resume_head_split --output RESUME_HEAD.json
launch rc_resume_geometry_full -m research_realized_coverage_v1.train_geometry --arm ordinary --name resume_geometry_full --steps 100
launch rc_resume_geometry_half -m research_realized_coverage_v1.train_geometry --arm ordinary --name resume_geometry_split --steps 100 --stop-after 50
launch rc_resume_geometry_finish -m research_realized_coverage_v1.train_geometry --arm ordinary --name resume_geometry_split --steps 100 --resume
launch rc_resume_geometry_compare -m research_realized_coverage_v1.resume_audit --names resume_geometry_full resume_geometry_split --output RESUME_GEOMETRY.json
