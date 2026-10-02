#!/usr/bin/env bash
# Independent seed-282000 batch; source is this immutable release directory.
# Do not alter or resume the original seed-281000 / 6c collection with this file.
set -euo pipefail
root=/home/wzy/dpvlm/route_set_v1
release="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data="$root/data/observation_multitask_validated_five_v1"
run_id="${1:-observation_multitask_validated_five_v1}"
case "$run_id" in ''|*[!a-zA-Z0-9_-]*) echo 'Invalid run id' >&2; exit 2;; esac
run="$root/runs/$run_id"
mkdir "$run"
echo $$ > "$run/pid"
date -u +%FT%TZ > "$run/started_at"
echo running > "$run/status"
printf '%s\n' "$release" > "$run/source_release"
printf '%s\n' "$data" > "$run/data_root"
sha256sum "$release/scripts/observation_collect_multitask.py" "$release/scripts/observation_collect_rlbench.py" "$release/routeset/multitask_fingerprints.py" "$release/configs/observation_multitask_validated_five_v1_registration.json" "${BASH_SOURCE[0]}" > "$run/source_sha256.txt"
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
cd "$release"
# Check only the five explicit prior pairs; push is intentionally left on.
"$root/.venv/bin/python" -c "import json; from pathlib import Path; r=Path('$root/runs'); paths=[r/'observation_demo_render_benchmark_v1/comparison.json']+[r/'observation_demo_render_five_task_audit_v1'/p/'comparison.json' for p in ['pick_and_lift_291000','take_lid_off_saucepan_311000','pick_up_cup_321000','slide_block_to_target_331000']]; assert all(json.loads(p.read_text())['single_parent_exact_equivalence'] for p in paths)"
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id registration --resume-strategy none -- \
    "$root/.venv-sim/bin/python" scripts/observation_collect_multitask.py --output "$data" --seed 282000 --camera-policy validated-five-v1 --schedule interleaved-early-dev-v1 --prepare-only
"$root/.venv/bin/python" -c "import json; from pathlib import Path; assert json.loads(Path('$data/partition_manifest.json').read_text())==json.loads(Path('$release/configs/observation_multitask_validated_five_v1_registration.json').read_text())"
"$root/.observation-deps/xorg/usr/bin/Xvfb" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$run/displaynum" >"$run/xvfb.log" 2>&1 &
xvfb_pid=$!
echo "$xvfb_pid" > "$run/xvfb_pid"
for attempt in $(seq 1 40); do
    [ -s "$run/displaynum" ] && break
    sleep .25
done
test -s "$run/displaynum"
export DISPLAY=":$(cat "$run/displaynum")"
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id registered_parents --resume-strategy flag -- \
    "$root/.venv-sim/bin/python" -u scripts/observation_collect_multitask.py --output "$data" --seed 282000 --camera-policy validated-five-v1 --schedule interleaved-early-dev-v1 --resume
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id cross_batch_layout_audit --resume-strategy fresh-output -- \
    "$root/.venv/bin/python" scripts/audit_multitask_layout_hashes.py --source "$data" --source "$root/data/observation_multitask_resume_probe_v1" --output "$run/cross_batch_layout_usage_gate.json"
