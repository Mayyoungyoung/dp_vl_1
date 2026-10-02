#!/usr/bin/env bash
# Independent TRAIN-only camera audit. Never changes an existing collection.
set -euo pipefail
root=/home/wzy/dpvlm/route_set_v1
release=/home/wzy/dpvlm/route_set_v1/research_v2/releases/7cb3aedef34b665c923b7427c1623867479fcd6e
run="$root/runs/observation_demo_render_push_on_control_v1"
mkdir "$run"
echo $$ > "$run/pid"
echo running > "$run/status"
date -u +%FT%TZ > "$run/started_at"
sha256sum "$release/scripts/benchmark_demo_render.py" "$release/scripts/observation_collect_multitask.py" "$release/scripts/observation_collect_rlbench.py" "${BASH_SOURCE[0]}" > "$run/source_sha256.txt"
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
export CODE_COMMIT="$(basename "$release")"
export PYTHONPATH="$release:$release/scripts:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="$root/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export CUDA_VISIBLE_DEVICES=''
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe LP_NUM_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
cd "$release"
echo 1 > "$run/additional_proposals_requested"
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id repeat_on --resume-strategy none -- \
    "$root/.venv-sim/bin/python" -u scripts/benchmark_demo_render.py --parent-id push_button_301000 --mode on --output "$run/repeat_on"
original="$root/runs/observation_demo_render_five_task_audit_v1/push_button_301000"
for mode in on off; do
    "$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id "original_${mode}_vs_repeat_on" --resume-strategy none -- \
        "$root/.venv-sim/bin/python" -u scripts/benchmark_demo_render.py --compare "$original/$mode" "$run/repeat_on" --output "$run/original_${mode}_vs_repeat_on.json"
done
