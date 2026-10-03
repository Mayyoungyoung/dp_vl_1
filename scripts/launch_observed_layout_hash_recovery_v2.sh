#!/usr/bin/env bash
# Explicit, separate stages. No automatic collection or certification chain.
set -euo pipefail
[[ $# -ge 2 && $# -le 3 ]] || exit 2
REV="$1"; STAGE="$2"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$STAGE" =~ ^(tests|prepare|recover|resume)$ ]] || exit 2
if [[ "${LAYOUT_HASH_RECOVERY_CPU_BOUND:-0}" != 1 ]]; then
  export LAYOUT_HASH_RECOVERY_CPU_BOUND=1
  exec taskset -c 3 bash "${BASH_SOURCE[0]}" "$@"
fi
P=/home/wzy/dpvlm/route_set_v1
SRC="$P/research_v2/releases/$REV"
OUT="$P/runs/observed_layout_variation_hash_recovery_v2"
DATA="$P/data/observed_layout_variation_hash_recovery_v2"
PY="$P/.venv/bin/python"
SCRIPT="$SRC/scripts/collect_observed_layout_hash_recovery.py"
[[ "$(readlink -f "${BASH_SOURCE[0]}")" == "$SRC/scripts/launch_observed_layout_hash_recovery_v2.sh" ]] || exit 2
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES=''
unset RESEARCH_GPU_UUID
RUNID="$STAGE"
if [[ "$STAGE" == resume ]]; then
  [[ $# -eq 3 && "$3" =~ ^recover_resume_[a-zA-Z0-9_]+$ ]] || exit 2
  RUNID="$3"
else
  [[ $# -eq 2 ]] || exit 2
fi
mkdir -p "$OUT"
[[ ! -e "$OUT/$RUNID.status.json" && ! -e "$OUT/$RUNID.lock" && ! -e "$OUT/$RUNID.identity.json" ]] || exit 2
"$PY" - "$SRC" "$OUT" "$REV" "$STAGE" "$RUNID" <<'PY'
import datetime,hashlib,json,os,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
src,out,revision,stage,runid=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
if src.name!=revision or sorted(os.sched_getaffinity(0))!=[3] or os.environ['CUDA_VISIBLE_DEVICES']!='':raise ValueError('Immutable source/CPU3/hiddenGPU required')
from scripts import collect_observed_layout_hash_recovery as scientific
extra=['scripts/launch_observed_layout_hash_recovery_v2.sh','tests/test_layout_hash_recovery.py','tests/test_observed_layout_variation.py','tests/fixtures/layout_hash_readback12.json']
sources=scientific.source_hashes();sources.update({p:hashlib.sha256((src/p).read_bytes()).hexdigest() for p in extra})
if stage!='tests':
 status=json.loads((out/'tests.status.json').read_text());identity=json.loads((out/'tests.identity.json').read_text())
 if status.get('exit_code')!=0 or status.get('status')!='completed' or status.get('code_commit')!=revision or identity['source_sha256']!=sources:raise ValueError('Actual same-source test success required')
 root=ET.parse(out/'tests.xml').getroot();cases=root.findall('.//testcase')
 if len(cases)!=48 or len([c for c in cases if c.attrib.get('classname','').endswith('test_layout_hash_recovery')])!=27 or any(c.find(x) is not None for c in cases for x in ('failure','error','skipped')):raise ValueError('Exact48/new27 tests with zero failures/skips required')
if stage in ('recover','resume'):
 status=json.loads((out/'prepare.status.json').read_text())
 if status.get('exit_code')!=0 or status.get('code_commit')!=revision:raise ValueError('Independent prepare stage must complete first')
 if (out/'recovery/complete.json').exists():raise ValueError('Already completed; no second recovery')
identity=dict(code_commit=revision,stage=stage,run_id=runid,source_sha256=sources,cpu_affinity=[3],cuda_visible_devices='',automatic_next_stage=False,automatic_certify=False,
             pre_display_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pre_display_monotonic_ns=time.monotonic_ns(),
             resume_command=['bash',str(src/'scripts/launch_observed_layout_hash_recovery_v2.sh'),revision,'resume','recover_resume_EXPLICIT_NEW_ID'])
(out/(runid+'.identity.json')).write_text(json.dumps(identity,indent=2)+'\n')
PY
record=("$PY" -m scripts.record_job --output "$OUT" --run-id "$RUNID" --resume-strategy none --)
if [[ "$STAGE" == tests ]]; then
  exec "${record[@]}" "$PY" -m pytest tests/test_layout_hash_recovery.py tests/test_observed_layout_variation.py -q --junitxml="$OUT/tests.xml"
elif [[ "$STAGE" == prepare ]]; then
  exec "${record[@]}" "$PY" "$SCRIPT" prepare --output "$DATA"
fi

# Only this directly spawned display is cleaned up. Other tasks are untouched.
DISPLAY_ROOT="$OUT/${RUNID}_display"
mkdir "$DISPLAY_ROOT"
mkdir "$DISPLAY_ROOT/xdg"; chmod 700 "$DISPLAY_ROOT/xdg"
SIM="$P/.observation-deps/CoppeliaSim"
XVFB="$P/.observation-deps/xorg/usr/bin/Xvfb"
[[ -f "$XVFB" && -d "$SIM" && -x "$P/.venv-sim/bin/python" ]] || exit 2
export COPPELIASIM_ROOT="$SIM" LD_LIBRARY_PATH="$SIM:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$SIM" LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe XDG_RUNTIME_DIR="$DISPLAY_ROOT/xdg"
"$XVFB" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$DISPLAY_ROOT/displaynum" >"$DISPLAY_ROOT/xvfb.log" 2>&1 &
XVFB_PID=$!
cleanup() {
  local original_status=$? xvfb_exit=0 child owned_running=0
  for child in $(jobs -pr); do [[ "$child" == "$XVFB_PID" ]] && owned_running=1; done
  if [[ "$owned_running" == 1 ]]; then
    kill "$XVFB_PID" 2>/dev/null || true
    for attempt in {1..20}; do kill -0 "$XVFB_PID" 2>/dev/null || break; sleep .25; done
    # Only kill again while this PID is still a running child of this shell.
    for child in $(jobs -pr); do [[ "$child" == "$XVFB_PID" ]] && kill -KILL "$XVFB_PID" 2>/dev/null || true; done
  fi
  wait "$XVFB_PID" 2>/dev/null || xvfb_exit=$?
  "$PY" - "$OUT" "$RUNID" "$XVFB_PID" "$xvfb_exit" "$original_status" <<'PY'
import datetime,json,sys,time
from pathlib import Path
out=Path(sys.argv[1]);runid=sys.argv[2];identity=json.loads((out/(runid+'.identity.json')).read_text())
now=time.monotonic_ns();receipt=dict(xvfb_pid=int(sys.argv[3]),xvfb_wait_status=int(sys.argv[4]),wrapper_exit_code=int(sys.argv[5]),
    end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pre_display_through_cleanup_seconds=(now-identity['pre_display_monotonic_ns'])/1e9,
    cost_scope='This wall interval includes display startup, record_job, child workers and cleanup; do not add to nested worker or record_job time. Initial source/test guards precede this interval.',
    cleanup_scope='Only directly owned shell background child',automatic_next_stage=False)
(out/(runid+'.display_wall_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
PY
  return "$original_status"
}
trap cleanup EXIT
printf '%s\n' "$XVFB_PID" >"$DISPLAY_ROOT/xvfb.pid"
sha256sum "$XVFB" >"$DISPLAY_ROOT/xvfb.sha256"
for attempt in {1..40}; do
  kill -0 "$XVFB_PID" || exit 2
  [[ -s "$DISPLAY_ROOT/displaynum" ]] && break
  sleep .25
done
DISPLAY_NUMBER="$(cat "$DISPLAY_ROOT/displaynum")"
[[ "$DISPLAY_NUMBER" =~ ^[0-9]+$ ]] || exit 2
export DISPLAY=":$DISPLAY_NUMBER"
args=(recover --corpus "$DATA" --run-root "$OUT" --sim-python "$P/.venv-sim/bin/python")
[[ "$STAGE" == resume ]] && args+=(--resume)
# Xvfb startup is outside this stage's record_job; identity/display receipts and
# shell caller wall time report it separately, not as inherited worker budget.
"${record[@]}" "$PY" "$SCRIPT" "${args[@]}"
