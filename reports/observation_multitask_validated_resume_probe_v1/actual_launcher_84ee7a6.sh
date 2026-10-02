#!/usr/bin/env bash
# Run this file from an immutable exported release. No formal batch is started.
set -euo pipefail
release=/home/wzy/dpvlm/route_set_v1/research_v2/releases/84ee7a6bb00a151bdbb2ebc34d2ba3b60b407320
root=/home/wzy/dpvlm/route_set_v1
run="$root/runs/observation_multitask_validated_resume_probe_v1"
data="$root/data/observation_multitask_validated_five_v1"
mkdir "$run"
echo $$ > "$run/pid"
date -u +%FT%TZ > "$run/started_at"
echo running > "$run/status"
printf '%s\n' "$release" > "$run/source_release"
sha256sum "$release/scripts/observation_collect_multitask.py" "$release/scripts/observation_collect_rlbench.py" "$release/routeset/multitask_fingerprints.py" "${BASH_SOURCE[0]}" > "$run/source_sha256.txt"
xvfb_pid=''
finish() {
    result=$?
    echo "$result" > "$run/exit_code"
    date -u +%FT%TZ > "$run/finished_at"
    if [ "$result" = 0 ]; then echo complete > "$run/status"; else echo failed > "$run/status"; fi
    if [ -n "$xvfb_pid" ] && [ "$(ps -o ppid= -p "$xvfb_pid" 2>/dev/null | tr -d ' ')" = "$$" ]; then kill "$xvfb_pid" 2>/dev/null || true; fi
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
test ! -e "$data/TRAIN/parents/reach_target_282000/attempts/slot_00/record.json"
"$root/.venv-sim/bin/python" -u scripts/observation_collect_multitask.py --output "$data" --seed 282000 --camera-policy validated-five-v1 --schedule interleaved-early-dev-v1 --prepare-only
set +e
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id first_slot --resume-strategy none -- \
    "$root/.venv-sim/bin/python" -u scripts/observation_collect_multitask.py --output "$data" --seed 282000 --camera-policy validated-five-v1 --schedule interleaved-early-dev-v1 --worker-parent reach_target_282000 --max-new-attempts 1
first_code=$?
set -e
if [ "$first_code" != 3 ]; then
    echo "First worker expected controlled exit 3; got $first_code" >&2
    exit 1
fi
parent="$data/TRAIN/parents/reach_target_282000"
sha256sum "$parent/attempts/slot_00/record.json" "$parent/mechanical_fingerprint.json" "$data/partition_manifest.json" "$data/source_manifest.json" > "$run/first_slot_before.sha256"
"$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id same_parent_resume --resume-strategy none -- \
    "$root/.venv-sim/bin/python" -u scripts/observation_collect_multitask.py --output "$data" --seed 282000 --camera-policy validated-five-v1 --schedule interleaved-early-dev-v1 --worker-parent reach_target_282000
sha256sum -c "$run/first_slot_before.sha256" > "$run/first_slot_preserved.txt"
export COLLECTION_PROBE_DATA="$data" COLLECTION_PROBE_RUN="$run"
"$root/.venv-sim/bin/python" - <<'PY'
import json,os,pathlib
from scripts.observation_collect_multitask import completed_records,export_role,mechanical_summary
root=pathlib.Path(os.environ['COLLECTION_PROBE_DATA']);run=pathlib.Path(os.environ['COLLECTION_PROBE_RUN'])
parent=root/'TRAIN/parents/reach_target_282000'
check=json.loads((parent/'latest_reconstruction_check.json').read_text())
assert check['world_difference']['max_abs']==0 and check['world_difference']['same_object_inventory']
assert not any(check['observation_max_differences'].values()) and check['language_equal']
records=completed_records(parent)
assert [row['attempt'] for row in records]==[0,1,2]
for row in records:
    restore=row['restore']
    assert restore['max_abs']==0 and restore['same_object_inventory'] and restore['language_equal']
    assert not any(restore['observed_field_max_difference'].values())
    assert row['motion_camera_policy']=='validated-five-v1'
    if row['demo_render_audit'] is not None:
        assert row['demo_render_audit']['rgbd_suppressed'] and row['demo_render_audit']['flags_restored']
        assert row['demo_render_audit']['initial_and_restore_rgbd_on']
plan=json.loads((root/'partition_manifest.json').read_text());export_role(root,'TRAIN',plan)
status=mechanical_summary(root,plan)
result=dict(parent_id='reach_target_282000',cross_process_reconstruction=check,completed_attempts=3,
            successes=sum(row['success'] for row in records),original_first_slot_preserved=True,
            all_three_strict_restores_passed=True,worker_sessions=[json.loads(p.read_text()) for p in sorted((parent/'sessions').glob('*.json'))],
            cumulative_worker_wall_seconds=status['parents'][0]['finalized_worker_elapsed_seconds'],
            full_motion_collision_certified=False,unique_valid_route_types=None,
            motion_camera_policy=plan['motion_camera_policy'],schedule=plan['schedule'],
            all_three_demo_contexts_verified=all(row['demo_render_audit'] is not None for row in records),
            attempt_render_audits=[dict(attempt=row['attempt'],success=row['success'],render=row['demo_render_audit'],error=row.get('error')) for row in records],
            persisted_registration_source_and_fingerprint_preserved=True,
            layout_usage_gate=json.loads((root/'layout_usage_gate.json').read_text()))
(run/'resume_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({key:result[key] for key in ['parent_id','successes','all_three_strict_restores_passed','all_three_demo_contexts_verified','cumulative_worker_wall_seconds','motion_camera_policy']}),flush=True)
PY
