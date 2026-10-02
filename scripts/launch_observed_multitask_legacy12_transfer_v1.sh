#!/usr/bin/env bash
# Execute from a reviewed immutable release; no training or checkpoint selection.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Reviewed immutable code commit required}"
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export TRANSFER_LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=-1
"$P/.venv/bin/python" - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
from routeset.common import sha256,write_json
from routeset.legacy_multitask_transfer import validate_checkpoint_plan,verify_export,validate_cache_compatibility
from scripts.evaluate_legacy_multitask_transfer import inspect_checkpoint

p=Path('/home/wzy/dpvlm/route_set_v1');src=Path.cwd()
assert src.name==os.environ['CODE_COMMIT'] and len(src.name)==40
config_path=src/'configs/observed_multitask_legacy12_transfer_v1.json'
config=json.loads(config_path.read_text())
assert config['execution_authorized'] and not config['checkpoint_reselection'] and config['train_steps']==0
plan=validate_checkpoint_plan(config)
d=p/'data'/config['output_dataset'];out=p/'runs'/config['output_family'];cache=d/'qwen_cache'
assert not d.exists() and not out.exists(),'Fresh pipeline only; inspect recorded failed stage and use its exact recovery command deliberately.'
assert len(os.sched_getaffinity(0))==1,'Caller must taskset to one authorized CPU.'
out.mkdir(parents=True)
write_json(out/'preregistered_config.json',config)
sources=[Path(os.environ['TRANSFER_LAUNCHER']),config_path,src/'scripts/record_job.py',
    src/'scripts/export_legacy_multitask_dev.py',src/'scripts/evaluate_legacy_multitask_transfer.py',
    src/'routeset/legacy_multitask_transfer.py',src/'scripts/observation_cache_qwen.py',
    src/'scripts/snapshot_multitask_observations.py',src/'routeset/multitask_fingerprints.py']
write_json(out/'source_hashes.json',{str(path):sha256(path) for path in sources})
write_json(out/'runtime.json',dict(cpu_affinity=sorted(os.sched_getaffinity(0)),omp_threads=os.environ['OMP_NUM_THREADS'],
    code_commit=src.name,gpu_uuid=os.environ['RESEARCH_GPU_UUID'],gpu_memory_fraction=.35,
    recovery='Stages have separate record_job commands. No automatic pipeline replay; inspect status and preserve outputs. Cache can resume identical hashes; evaluation requires fresh output.'))
def record(run_id,python,module,options,resume='fresh-output',gpu=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1' if gpu else '-1')
    command=[sys.executable,'-m','scripts.record_job','--output',str(out),'--run-id',run_id,'--resume-strategy',resume,'--',
        str(p/python),'-m',module]+options
    subprocess.run(command,check=True,env=env)
record('source_tests','.venv/bin/python','pytest',['-q','tests/test_legacy_multitask_transfer.py','tests/test_legacy_multitask_transfer_torch.py'],resume='none')
configs=[];audits=[]
for row in plan:
    c,saved,source=inspect_checkpoint(row);configs.append(c);audits.append(source);del saved
write_json(out/'original_model_source_audits.json',audits)
record('export','.venv/bin/python','scripts.export_legacy_multitask_dev',['--registration',str(config_path),'--output',str(d)])
export,gate=verify_export(d,config)
assert export['requested_parents']==12 and export['requested_attempts']==36
assert 0<export['observations']<=48,'One complete cache, at most48 registered language inputs; never truncate.'
write_json(out/'pre_cache_gate.json',gate)
# The unchanged cache producer and model source are checked above. Actual output configuration is rechecked below.
expected=dict(configs[0]['cache_config'],manifest_sha256=sha256(d/'observations.jsonl'))
validate_cache_compatibility(expected,configs,sha256(d/'observations.jsonl'))
record('qwen_cache','.venv-qwen/bin/python','scripts.observation_cache_qwen',[
    '--model',str(p/'data/qwen3-vl-2b-instruct-89644892'),'--manifest',str(d/'observations.jsonl'),
    '--output',str(cache),'--device','cuda','--threads','1','--gpu-memory-fraction','0.35',
    '--max-pixels',str(config['cache']['max_pixels'])],resume='none',gpu=True)
cache_config=json.loads((cache/'cache_config.json').read_text());cache_status=json.loads((cache/'status.json').read_text())
compatibility=validate_cache_compatibility(cache_config,configs,sha256(d/'observations.jsonl'))
assert cache_status['samples']==export['observations'] and cache_status['status']=='frozen_rgb_language_feature_extraction_complete'
records=[json.loads(line) for line in (cache/'samples.jsonl').read_text().splitlines()]
assert len(records)==len({r['id'] for r in records})==export['observations']
for row in records:assert sha256(cache/row['file'])==row['sha256']
write_json(out/'actual_cache_receipt.json',dict(compatibility=compatibility,status=cache_status,
    cache_config_sha256=sha256(cache/'cache_config.json'),samples_sha256=sha256(cache/'samples.jsonl'),
    sample_count=len(records),cache_gpu_hours_reserved=cache_status['total_seconds']/3600,
    scope='Actual full RGB+language encoding, once for each new input. Shared feature caching is not end-to-end request latency.'))
print('Actual Qwen caching completed; GPU released before12 CPU checkpoint evaluations.',flush=True)
record('frozen_transfer','.venv/bin/python','scripts.evaluate_legacy_multitask_transfer',[
    '--registration',str(config_path),'--data',str(d),'--output',str(out/'evaluation')])
verify_export(d,config)
print('All twelve fixed checkpoint transfers completed; no training or selection performed.',flush=True)
PY
