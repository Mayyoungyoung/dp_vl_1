#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=0f1d5bfbd4ff788a6ea311339adfe77449db9634
SRC="$P/research_v2/releases/$REV"
BASE="$P/runs/observed_two_row_diffusion_v1"
STAGE="${1:?train, repeat1, repeat2, fixed-last-train or denoising-diagnostic}"
ARM="${2:?independent or set}"
MODE="${3:?train pause2/resume/full; all other stages fresh}"
[[ "$ARM" == independent || "$ARM" == set ]] || exit 2
TARGET="$BASE/$ARM"
RUN="${ARM}_${STAGE}_${MODE}"
EXTRA=()
case "$STAGE" in
 train)
  case "$MODE" in
   pause2) EXTRA+=(--stop-after 2);;
   resume) EXTRA+=(--resume); RUN="${RUN}_$(date -u +%Y%m%dT%H%M%SZ)_$$";;
   full) ;;
   *) exit 2;;
  esac
  ;;
 repeat1|repeat2|fixed-last-train|denoising-diagnostic)
  [[ "$MODE" == fresh ]] || exit 2
  EXTRA+=(--train-run "$TARGET")
  TARGET="$BASE/${ARM}_${STAGE}"
  ;;
 *) exit 2;;
esac
export DIFFUSION_TRAIN_WRAPPER="$(readlink -f "${BASH_SOURCE[0]}")"
export DIFFUSION_STAGE="$STAGE" DIFFUSION_ARM="$ARM" DIFFUSION_MODE="$MODE" DIFFUSION_RUN_ID="$RUN" DIFFUSION_TARGET="$TARGET"
cd "$SRC"
export CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/tests" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" - <<'PY'
import hashlib,json,os,subprocess
from pathlib import Path
p=Path('/home/wzy/dpvlm/route_set_v1');revision='0f1d5bfbd4ff788a6ea311339adfe77449db9634'
source=p/'research_v2/releases'/revision;base=p/'runs/observed_two_row_diffusion_v1'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
if Path.cwd()!=source or os.environ.get('CODE_COMMIT')!=revision or sorted(os.sched_getaffinity(0))!=[1]:
 raise ValueError('Immutable0f1d5bf release and exactly CPU1 required')
uuid=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
if uuid!=os.environ['RESEARCH_GPU_UUID']:raise ValueError('PhysicalGPU1 identity differs')
validation=p/'runs/observed_diffusion_validation_v1'/revision
if sha(validation/'validation_receipt.json')!='016493e78ff5b8a129616189ed4e933dd0dfcf4cd1023fe3990fea8342177dde':
 raise ValueError('Actual passed35 validation receipt changed')
receipt=json.loads((validation/'validation_receipt.json').read_text())
identity=json.loads((validation/'validation_identity.json').read_text())
status=json.loads((validation/'tests.status.json').read_text())
if (receipt['status']!='passed' or receipt['junit']['tests']!=35 or receipt['junit']['skipped']!=0 or
 receipt['junit']['real_torch_core']!=17 or receipt['junit']['real_torch_driver']!=5 or
 status['status']!='completed' or status['exit_code']!=0 or status['code_commit']!=revision or
 receipt['identity_sha256']!=sha(validation/'validation_identity.json') or
 receipt['pytest_xml_sha256']!=sha(validation/'pytest.xml')):
 raise ValueError('Required actual CPU/Torch validation did not pass')
for name,value in identity['source_sha256'].items():
 if sha(source/name)!=value:raise ValueError('Validated implementation changed: '+name)
stage,arm,mode,run=(os.environ['DIFFUSION_'+key] for key in ('STAGE','ARM','MODE','RUN_ID'))
target=Path(os.environ['DIFFUSION_TARGET']);original=base/arm
if target.parent!=base or (target/'active.lock').exists():raise ValueError('Unexpected output or active process')
if stage=='train' and mode=='resume':
 if not (target/'last.pt').is_file() or not (target/'config.json').is_file() or (target/'summary.json').exists():
  raise ValueError('Resume requires incomplete original checkpoint; never resume completed training')
elif target.exists() and any(target.iterdir()):raise ValueError('Fresh output required; preserve all prior attempts')
if stage!='train':
 summary=json.loads((original/'summary.json').read_text())
 if summary['status']!='completed' or summary['last_step']!=12000 or summary['arm']!=arm:
  raise ValueError('Independent diagnostic requires completed matching arm')
wrapper=Path(os.environ['DIFFUSION_TRAIN_WRAPPER']).resolve()
if wrapper!=p/'research_v2/incoming/observed_diffusion_train_0f1d5bf.sh':
 raise ValueError('Only the reviewed immutable incoming wrapper is supported')
jobs=base/'job_records';jobs.mkdir(parents=True,exist_ok=True)
if (jobs/(run+'.status.json')).exists() or (jobs/(run+'.lock')).exists():raise ValueError('Do not overwrite an issued job')
launcher=dict(path=str(wrapper),sha256=sha(wrapper),source_commit=revision)
saved=base/'training_wrapper_identity.json'
if saved.exists():
 if json.loads(saved.read_text())!=launcher:raise ValueError('Training wrapper changed across stages')
else:saved.write_text(json.dumps(launcher,indent=2)+'\n')
preflight=dict(launcher=launcher,stage=stage,arm=arm,mode=mode,output=str(target),run_id=run,
 cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_uuid=uuid,gpu_memory_fraction=.35,
 validation_receipt_sha256=sha(validation/'validation_receipt.json'),automatic_stage_chaining=False,
 resume_command=['taskset','-c','1','bash',str(wrapper),'train',arm,'resume'] if stage=='train' else None,
 recovery_note='Administrative pause2 resumes without --stop-after. Hard-crash issued work must match checkpoint or complete sealed DEV pool; no automatic replay/retry.')
with (jobs/(run+'.preflight.json')).open('x') as f:f.write(json.dumps(preflight,indent=2)+'\n')
print(json.dumps(preflight),flush=True)
PY
exec "$P/.venv/bin/python" -m scripts.record_job --output "$BASE/job_records" --run-id "$RUN" --resume-strategy none -- \
 "$P/.venv/bin/python" -m scripts.train_observed_two_row_diffusion \
 --stage "$STAGE" --arm "$ARM" --config "$SRC/configs/observed_two_row_diffusion_v1.json" \
 --data "$P/data/observation_two_row_composite108_v1" \
 --quality-audit "$P/runs/observed_two_row_composite108_preparation_v1/train_quality.json" \
 --ordinary-run "$P/runs/observed_two_row_composite108_v1/peak_seed0" \
 --output "$TARGET" "${EXTRA[@]}"
