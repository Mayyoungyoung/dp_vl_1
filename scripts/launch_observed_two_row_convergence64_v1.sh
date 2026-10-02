#!/usr/bin/env bash
# stage1500 and finish are separate launches; never automatically continue.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Reviewed immutable source commit required}"
STAGE="${2:?stage1500 or finish required}"
RECOVERY="${3:-fresh}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$STAGE" == stage1500 || "$STAGE" == finish ]] || exit 2
[[ "$RECOVERY" == fresh || "$RECOVERY" == resume ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONPATH="$SRC" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export CONVERGENCE_LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
"$P/.venv/bin/python" - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
assert len(os.sched_getaffinity(0))==1,'One authorized CPU is required'
assert Path.cwd().name==os.environ['CODE_COMMIT']
assert subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==os.environ['RESEARCH_GPU_UUID']
out=Path('/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_prefix76_convergence_v1')
out.mkdir(parents=True,exist_ok=True)
p=Path(os.environ['CONVERGENCE_LAUNCHER']);value=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),code_commit=os.environ['CODE_COMMIT'],cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_uuid=os.environ['RESEARCH_GPU_UUID'],gpu_memory_fraction=.35)
record=out/'launcher_identity.json'
if record.exists():assert json.loads(record.read_text())==value,'Frozen launcher changed'
else:record.write_text(json.dumps(value,indent=2)+'\n')
PY
EXTRA=()
if [[ "$RECOVERY" == resume ]]; then EXTRA+=(--resume); fi
RUN_ID="$STAGE"
if [[ "$RECOVERY" == resume ]]; then RUN_ID="${STAGE}_resume_$(date -u +%Y%m%dT%H%M%SZ)_$$"; fi
OUT="$P/runs/observed_two_row_prefix76_convergence_v1"
[[ ! -e "$OUT/$RUN_ID.status.json" ]] || { echo 'Preserve completed/failed stage status; use explicit individual resume only.' >&2; exit 2; }
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id "$RUN_ID" --resume-strategy flag -- \
  "$P/.venv/bin/python" -m scripts.train_observed_two_row_convergence64 --stage "$STAGE" --device cuda "${EXTRA[@]}"
