#!/usr/bin/env bash
set -euo pipefail
source_commit=1915de7dbe2c8cbc772250e092be87c83c7df786
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'Missing authorized fixed release SHA' >&2; exit 2; }
root=/home/wzy/dpvlm/route_set_v1
release="$root/research_v2/releases/$source_commit"
run="$root/runs/observed_two_row_static_contacts_v1"
data="$root/data/observed_two_row_static_contacts_v1"
test -d "$release"
test ! -e "$data"
test ! -e "$run"
mkdir "$run"
echo $$ > "$run/pid"
echo running > "$run/status"
date -u +%FT%TZ > "$run/started_at"
printf '%s\n' "$release" > "$run/source_release"
printf '%s\n' "$data" > "$run/data_root"
cp "${BASH_SOURCE[0]}" "$run/frozen_job.sh"
sha256sum "$release/scripts/diagnose_two_row_static_contacts.py" "$release/scripts/diagnose_two_row_endpoint_ik.py" "$release/scripts/collect_observed_two_row_pilot.py" "$release/scripts/two_row_anchor.py" "$release/scripts/collect_obstacle_reach.py" "$release/scripts/observation_collect_rlbench.py" "$release/configs/observed_two_row_static_contacts_v1.json" "$release/tests/test_two_row_endpoint_ik.py" "$release/tests/test_two_row_static_contacts.py" "$release/tests/test_two_row_anchor.py" "$release/tests/test_observed_two_row_pilot.py" "${BASH_SOURCE[0]}" > "$run/source_sha256.txt"
xvfb_pid=''
finish() {
    result=$?
    echo "$result" > "$run/exit_code"
    date -u +%FT%TZ > "$run/finished_at"
    if [ "$result" = 0 ]; then echo complete > "$run/status"; else echo failed > "$run/status"; fi
    if [ -n "$xvfb_pid" ] && [ "$(ps -o ppid= -p "$xvfb_pid" 2>/dev/null | tr -d ' ')" = "$$" ]; then
        kill "$xvfb_pid" 2>/dev/null || true
    fi
}
trap finish EXIT
export CODE_COMMIT="$source_commit"
export PYTHONPATH="$release:$release/scripts:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="$root/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export CUDA_VISIBLE_DEVICES=''
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe LP_NUM_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$release"
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id targeted_tests --resume-strategy none -- \
    "$root/.venv/bin/python" -m pytest -q tests/test_observed_two_row_pilot.py tests/test_two_row_anchor.py tests/test_two_row_endpoint_ik.py tests/test_two_row_static_contacts.py
export XDG_RUNTIME_DIR="$run/xdg"
mkdir "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
"$root/.observation-deps/xorg/usr/bin/Xvfb" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$run/displaynum" >"$run/xvfb.log" 2>&1 &
xvfb_pid=$!
echo "$xvfb_pid" > "$run/xvfb_pid"
for attempt in $(seq 1 40); do
    [ -s "$run/displaynum" ] && break
    sleep .25
done
test -s "$run/displaynum"
export DISPLAY=":$(cat "$run/displaynum")"
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id static_contacts --resume-strategy none -- \
    "$root/.venv-sim/bin/python" -u scripts/diagnose_two_row_static_contacts.py \
    --config "$release/configs/observed_two_row_static_contacts_v1.json" --output "$data"
