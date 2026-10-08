#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job strong_counterfactual_tests timeout 60 "$PY" -m pytest -q tests/test_research_v3_counterfactual.py
job strong_counterfactual_v1 timeout 120 "$PY" -m scripts.research_v3_counterfactual --output strong_counterfactual_v1 --strong-only
