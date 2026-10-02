#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
run_id=observation_qwen_download_20261002
run="runs/$run_id"
mkdir -p "$run"
exec 9>"$run/lock"
flock -n 9 || exit 73
echo $$ > "$run/pid"
date -u +%FT%TZ > "$run/started_at"
echo running > "$run/status"
mkdir -p "$run/source"
cp scripts/observation_download_qwen.py scripts/observation_download_job.sh scripts/observation_model_manifest.json "$run/source/"
sha256sum "$run/source/"* > "$run/source_sha256.txt"
.venv/bin/python "$run/source/observation_download_qwen.py" --output data/qwen3-vl-2b-instruct-89644892 --endpoint https://hf-mirror.com > "$run/download.log" 2>&1
result=$?
echo "$result" > "$run/exit_code"
date -u +%FT%TZ > "$run/finished_at"
if [ "$result" = 0 ]; then echo complete > "$run/status"; else echo failed > "$run/status"; fi
exit "$result"
