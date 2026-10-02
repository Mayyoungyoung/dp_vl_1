#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Reviewed immutable commit required}"
STAGE="${2:?tests or audit required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$STAGE" == tests || "$STAGE" == audit ]] || exit 2
SRC="$P/research_v2/releases/$REVISION"
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
[[ "$SELF" == "$SRC/scripts/launch_two_row_route_conditioning_v1.sh" ]] || exit 2
cd "$SRC"
export PYTHONPATH="$SRC" CODE_COMMIT="$REVISION" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
OUT="$P/runs/observed_two_row_route_conditioning_v1"
export ROUTE_PROBE_OUT="$OUT" ROUTE_PROBE_STAGE="$STAGE" ROUTE_PROBE_SELF="$SELF"
"$P/.venv/bin/python" - <<'PY'
import hashlib,json,os
from pathlib import Path
assert len(os.sched_getaffinity(0))==1,'Caller must choose exactly one authorized CPU'
root=Path.cwd();out=Path(os.environ['ROUTE_PROBE_OUT']);stage=os.environ['ROUTE_PROBE_STAGE']
out.mkdir(parents=True,exist_ok=True)
names=['scripts/audit_two_row_route_conditioning.py','tests/test_two_row_route_conditioning.py','configs/two_row_route_conditioning_v1.json','scripts/launch_two_row_route_conditioning_v1.sh','reports/TWO_ROW_ROUTE_CONDITIONING_PROTOCOL.md']
identity=dict(source_commit=os.environ['CODE_COMMIT'],files={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})
if stage=='tests':
 assert not (out/'tests.status.json').exists() and not (out/'source_identity.json').exists(),'Fresh tests only'
 (out/'source_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
else:
 assert json.loads((out/'source_identity.json').read_text())==identity,'Source changed after tests'
 status=json.loads((out/'tests.status.json').read_text())
 assert status['status']=='completed' and status['exit_code']==0,'Actual CPU tests required'
 import xml.etree.ElementTree as E
 xml=E.parse(out/'tests.xml');cases=xml.findall('.//testcase')
 assert len(cases)>=14 and not xml.findall('.//skipped') and not xml.findall('.//failure') and not xml.findall('.//error'),'All actual Torch tests, no skips'
 assert not (out/'analysis').exists() and not (out/'audit.status.json').exists(),'Fresh-only no replay'
PY
if [[ "$STAGE" == tests ]]; then
 exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id tests --resume-strategy none -- \
   "$P/.venv/bin/python" -m pytest tests/test_two_row_route_conditioning.py -q --junitxml="$OUT/tests.xml"
fi
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" - <<'PY'
import subprocess
assert subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
PY
exec "$P/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id audit --resume-strategy none -- \
 "$P/.venv/bin/python" -m scripts.audit_two_row_route_conditioning --project "$P" \
 --policy "$SRC/configs/two_row_route_conditioning_v1.json" --output "$OUT/analysis" --device cuda
