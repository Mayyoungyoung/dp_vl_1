#!/usr/bin/env bash
set -euo pipefail
sha="${1:?immutable reviewed release SHA}"
action="${2:?tests, prepare, train32, train64, train128, train256, dev32}"
mode="${3:-resume}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] && [[ "$mode" = fresh || "$mode" = resume ]] || exit 2
case "$action" in tests|prepare|train32|train64|train128|train256|dev32) ;; *) exit 2;; esac
root=/home/wzy/dpvlm/route_set_v1
release="$root/research_v2/releases/$sha"
run="$root/runs/observed_two_row_extension288_v1"
corpus="$root/data/observed_two_row_extension288_v1"
test -d "$release"
mkdir -p "$run/sessions"
session="$run/sessions/$(date -u +%Y%m%dT%H%M%SZ)_${action}_$$"
mkdir "$session"
cp "${BASH_SOURCE[0]}" "$session/frozen_launcher.sh"
printf '%s\n' "$release" > "$session/source_release"
printf '%s\n' "$action" > "$session/action"
printf '%s\n' "$mode" > "$session/mode"
printf '%s\n' "$0 $*" > "$session/command"
echo $$ > "$session/pid"
date -u +%FT%TZ > "$session/started_at"
sha256sum "$release/scripts/collect_two_row_extension.py" "$release/scripts/register_two_row_extension.py" \
    "$release/scripts/collect_two_row_formal.py" "$release/scripts/collect_two_row_canonical_repair.py" \
    "$release/configs/observed_two_row_extension288_v1.json" "$release/configs/observed_two_row_extension288_exclusions_v1.json" \
    "$release/configs/observed_two_row_extension288_registered_v1.json" \
    "$release/scripts/run_two_row_extension_shard.sh" "${BASH_SOURCE[0]}" > "$session/source_sha256.txt"
export CODE_COMMIT="$sha" CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$release:$release/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$release"
finish_simple() {
    code=$?
    printf '%s\n' "$code" > "$session/exit_code"
    date -u +%FT%TZ > "$session/finished_at"
    if [ "$code" -eq 0 ]; then echo completed > "$session/status"; else echo failed > "$session/status"; fi
}
trap finish_simple EXIT
if [ "$action" = tests ]; then
    taskset -c 2 "$root/.venv/bin/python" scripts/record_job.py --output "$session" --run-id targeted_tests --resume-strategy none -- \
        "$root/.venv/bin/python" -m pytest -q tests/test_two_row_extension_registration.py tests/test_two_row_extension_collection.py \
        tests/test_two_row_formal_collection.py tests/test_two_row_formal_registration.py tests/test_two_row_canonical_repair.py \
        tests/test_two_row_layout4.py tests/test_two_row_layout4_analysis.py tests/test_observed_two_row_pilot.py \
        tests/test_two_row_anchor.py tests/test_two_row_endpoint_ik.py tests/test_two_row_static_contacts.py
    exit 0
fi
if [ "$action" = prepare ]; then
    test ! -e "$corpus"
    taskset -c 2 "$root/.venv/bin/python" scripts/record_job.py --output "$session" --run-id registration_prepare --resume-strategy none -- \
        "$root/.venv/bin/python" scripts/collect_two_row_extension.py prepare \
        --registration "$release/configs/observed_two_row_extension288_registered_v1.json" --output "$corpus"
    # Deliberately return. Preparation never launches simulation workers.
    exit 0
fi
test -f "$corpus/corpus_manifest.json"
taskset -c 2 "$root/.venv/bin/python" scripts/record_job.py --output "$session" --run-id stage_gate --resume-strategy none -- \
    "$root/.venv/bin/python" scripts/collect_two_row_extension.py check-stage --corpus "$corpus" --stage "$action"
taskset -c 2 bash "$release/scripts/run_two_row_extension_shard.sh" "$sha" 0 "$session" "$action" "$mode" >"$session/shard0_launch.log" 2>&1 &
pid0=$!
taskset -c 3 bash "$release/scripts/run_two_row_extension_shard.sh" "$sha" 1 "$session" "$action" "$mode" >"$session/shard1_launch.log" 2>&1 &
pid1=$!
printf '%s\n' "$pid0" > "$session/shard0_pid"
printf '%s\n' "$pid1" > "$session/shard1_pid"
code0=0;code1=0
wait "$pid0" || code0=$?
wait "$pid1" || code1=$?
printf '%s\n' "$code0" > "$session/shard0_exit_code"
printf '%s\n' "$code1" > "$session/shard1_exit_code"
if { [ "$code0" -ne 0 ] && [ "$code0" -ne 3 ]; } || { [ "$code1" -ne 0 ] && [ "$code1" -ne 3 ]; }; then exit 1; fi
if [ "$code0" -eq 3 ] || [ "$code1" -eq 3 ]; then
    trap - EXIT
    echo internal_budget_paused > "$session/status"
    echo 3 > "$session/exit_code"
    date -u +%FT%TZ > "$session/finished_at"
    exit 3
fi
