#!/usr/bin/env bash
# Five additional TRAIN-only on/off pairs; never changes a formal collection.
set -euo pipefail
root=/home/wzy/dpvlm/route_set_v1
release="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="$root/runs/observation_demo_render_five_task_audit_v1"
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
parents=(pick_and_lift_291000 push_button_301000 take_lid_off_saucepan_311000 pick_up_cup_321000 slide_block_to_target_331000)
printf '%s\n' "${parents[@]}" > "$run/parent_ids.txt"
echo 10 > "$run/additional_proposals_requested"
any_failed=0
for parent in "${parents[@]}"; do
    mkdir "$run/$parent"
    for mode in on off; do
        if "$root/.venv/bin/python" scripts/record_job.py --output "$run/$parent" --run-id "camera_$mode" --resume-strategy none -- \
            "$root/.venv-sim/bin/python" -u scripts/benchmark_demo_render.py --parent-id "$parent" --mode "$mode" --output "$run/$parent/$mode"; then
            :
        else
            any_failed=1
        fi
    done
    if [ -f "$run/$parent/on/result.json" ] && [ -f "$run/$parent/off/result.json" ]; then
        if "$root/.venv/bin/python" scripts/record_job.py --output "$run/$parent" --run-id comparison --resume-strategy none -- \
            "$root/.venv-sim/bin/python" -u scripts/benchmark_demo_render.py --compare "$run/$parent/on" "$run/$parent/off" --output "$run/$parent/comparison.json"; then
            :
        else
            any_failed=1
        fi
    else
        echo missing_arm_result > "$run/$parent/comparison_not_run"
        any_failed=1
    fi
done
exit "$any_failed"
