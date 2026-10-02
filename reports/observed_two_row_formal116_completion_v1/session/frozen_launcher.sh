#!/usr/bin/env bash
set -euo pipefail
sha="${1:?immutable reviewed release SHA}"
mode="${2:-fresh}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] && [[ "$mode" = fresh || "$mode" = resume ]] || { echo "Invalid immutable SHA or mode" >&2; exit 2; }
root=/home/wzy/dpvlm/route_set_v1
release="$root/research_v2/releases/$sha"
run="$root/runs/observed_two_row_formal116_v1"
corpus="$root/data/observed_two_row_formal116_v1"
if [ "$mode" = fresh ]; then
    test ! -e "$run"
    test ! -e "$corpus"
    mkdir "$run"
else
    test -f "$corpus/corpus_manifest.json"
    test -d "$run"
fi
session="$run/sessions/$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$session"
cp "${BASH_SOURCE[0]}" "$session/frozen_launcher.sh"
printf '%s\n' "$release" > "$session/source_release"
printf '%s\n' "$mode" > "$session/mode"
echo $$ > "$session/pid"
date -u +%FT%TZ > "$session/started_at"
sha256sum "$release/scripts/collect_two_row_formal.py" "$release/scripts/collect_two_row_canonical_repair.py" "$release/scripts/register_two_row_formal.py" "$release/configs/observed_two_row_formal116_registered_v1.json" "$release/scripts/run_two_row_formal_shard.sh" "${BASH_SOURCE[0]}" > "$session/source_sha256.txt"
export CODE_COMMIT="$sha" CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$release:$release/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$release"
taskset -c 2 "$root/.venv/bin/python" scripts/record_job.py --output "$session" --run-id targeted_tests --resume-strategy none -- \
    "$root/.venv/bin/python" -m pytest -q tests/test_two_row_formal_collection.py tests/test_two_row_formal_registration.py tests/test_two_row_canonical_repair.py tests/test_two_row_layout4.py tests/test_two_row_layout4_analysis.py tests/test_observed_two_row_pilot.py tests/test_two_row_anchor.py tests/test_two_row_endpoint_ik.py tests/test_two_row_static_contacts.py
if [ "$mode" = fresh ]; then
    taskset -c 2 "$root/.venv/bin/python" scripts/record_job.py --output "$session" --run-id registration_prepare --resume-strategy none -- \
        "$root/.venv/bin/python" scripts/collect_two_row_formal.py prepare \
        --registration "$release/configs/observed_two_row_formal116_registered_v1.json" --output "$corpus"
fi
resume=false
if [ "$mode" = resume ]; then resume=true; fi
taskset -c 2 bash "$release/scripts/run_two_row_formal_shard.sh" "$sha" 0 "$session" "$resume" >"$session/shard0_launch.log" 2>&1 &
pid0=$!
taskset -c 3 bash "$release/scripts/run_two_row_formal_shard.sh" "$sha" 1 "$session" "$resume" >"$session/shard1_launch.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" > "$session/shard0_pid"
printf '%s\n' "$pid1" > "$session/shard1_pid"
code0=0;code1=0
wait "$pid0" || code0=$?
wait "$pid1" || code1=$?
printf '%s\n' "$code0" > "$session/shard0_exit_code"
printf '%s\n' "$code1" > "$session/shard1_exit_code"
date -u +%FT%TZ > "$session/finished_at"
if [ "$code0" -ne 0 ] || [ "$code1" -ne 0 ]; then echo failed > "$session/status"; exit 1; fi
echo completed > "$session/status"
