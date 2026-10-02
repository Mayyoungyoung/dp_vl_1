#!/usr/bin/env bash
set -uo pipefail
main() {
cd "$(dirname "$0")/.."
run_id="$1"
shift
run="runs/$run_id"
mkdir -p "$run"
exec 9>"$run/lock"
flock -n 9 || exit 73
echo $$ > "$run/pid"
date -u +%FT%TZ > "$run/started_at"
echo running > "$run/status"
mkdir -p "$run/source"
cache_script="${QWEN_CACHE_SCRIPT:-scripts/observation_cache_qwen.py}"
cp "$cache_script" scripts/observation_qwen_job.sh "$(dirname "$cache_script")/observation_model_manifest.json" "$run/source/"
sha256sum "$run/source/"* > "$run/source_sha256.txt"
if [ -n "${SOURCE_COMMIT:-}" ]; then printf "%s\n" "$SOURCE_COMMIT" > "$run/source_commit"; fi
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venv-qwen/bin/python -u "$run/source/observation_cache_qwen.py" \
    --model data/qwen3-vl-2b-instruct-89644892 --device cuda \
    "$@" > "$run/cache.log" 2>&1
result=$?
echo "$result" > "$run/exit_code"
date -u +%FT%TZ > "$run/finished_at"
if [ "$result" = 0 ]; then echo complete > "$run/status"; else echo failed > "$run/status"; fi
exit "$result"
}
main "$@"
