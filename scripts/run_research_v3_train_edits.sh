#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
bash scripts/launch_research_v3.sh --id train_edit_tests -- timeout 60 "$PY" -m pytest -q tests/test_research_v3_counterfactual.py tests/test_research_v3_witness_distance.py
bash scripts/launch_research_v3.sh --id train_edit_retention_v1 -- timeout 120 "$PY" -m scripts.research_v3_train_edit_retention --output train_edit_retention_v1
