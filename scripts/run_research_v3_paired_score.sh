#!/usr/bin/env bash
# Run only after the separately checked six-parent smoke collection.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/paired_score_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
"$PY" -c "import json; assert json.load(open('$P/runs/research_v3_v1/paired_score_collection_000_006/summary.json'))['passed']"
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job paired_score_collect_remaining timeout 2220 bash scripts/launch_research_v3_score_collect.sh collect --start 6 --stop 192
job paired_score_export "$PY" -m scripts.research_v3_score_data export
for role in SCORE_TRAIN DEV_SCORE CALIBRATION; do
  D="$P/data/research_v3_paired_score_v1/exports/$role"
  job "paired_score_qwen_$role" "$P/.venv-qwen/bin/python" -m scripts.observation_cache_qwen \
    --model "$P/data/qwen3-vl-2b-instruct-89644892" --manifest "$D/observations.jsonl" \
    --output "$D/qwen_cache" --device cuda --threads 4 --gpu-memory-fraction .35
done
for role in SCORE_TRAIN DEV_SCORE CALIBRATION paired_dev; do
  job "paired_score_pool_$role" "$PY" -m scripts.research_v3_matched_q pool --role "$role" --domain paired
done
job paired_score_roles "$PY" -m scripts.research_v3_matched_q roles --domain paired
for seed in 0 1 2; do
  job "paired_score_fit_$seed" "$PY" -m scripts.research_v3_matched_q fit --seed "$seed" --domain paired
  job "paired_score_calibrate_$seed" "$PY" -m scripts.research_v3_matched_q calibrate --seed "$seed" --domain paired
done
job paired_score_analysis "$PY" -m scripts.research_v3_matched_q analyze --domain paired
