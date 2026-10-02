#!/usr/bin/env bash
# Continue the registered corpus after the actual cross-process probe passed.
# Collector source stays 6c44469, including cameras-on motion recording.
set -euo pipefail
root=/home/wzy/dpvlm/route_set_v1
release="$root/research_v2/releases/6c4446921317fab43066d4928e7a5cdcbbf340d1"
data="$root/data/observation_multitask_resume_probe_v1"
run_id="${1:-observation_multitask_six_formal_v1}"
case "$run_id" in ''|*[!a-zA-Z0-9_-]*) echo 'Invalid run id' >&2; exit 2;; esac
run="$root/runs/$run_id"
test -f "$root/runs/observation_multitask_resume_probe_v1/resume_validation.json"
"$root/.venv/bin/python" -c "import json; from pathlib import Path; p=Path('$root/runs/observation_multitask_resume_probe_v1/resume_validation.json'); r=json.loads(p.read_text()); assert r['all_three_strict_restores_passed'] and r['original_first_slot_preserved']"
mkdir "$run"
echo $$ > "$run/pid"
date -u +%FT%TZ > "$run/started_at"
echo running > "$run/status"
printf '%s\n' "$release" > "$run/source_release"
printf '%s\n' "$data" > "$run/data_root"
sha256sum "$release/scripts/observation_collect_multitask.py" "$release/scripts/observation_collect_rlbench.py" "${BASH_SOURCE[0]}" > "$run/source_sha256.txt"
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
export CODE_COMMIT=6c4446921317fab43066d4928e7a5cdcbbf340d1
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
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id remaining_registered_parents --resume-strategy flag -- \
    "$root/.venv-sim/bin/python" -u scripts/observation_collect_multitask.py --output "$data" --resume
