#!/usr/bin/env bash
# Reviewed launch; execute only the copied immutable launcher/config.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION=6bc2b8d5e055c999c87fb37e5bcd9fa5d8237f40
SRC="$P/research_v2/releases/$REVISION"
CONFIG="${1:?Reviewed immutable config JSON path is required}"
cd "$SRC"
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CODE_COMMIT="$REVISION"
export MULTITASK_LAUNCH_CONFIG="$CONFIG" MULTITASK_LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
"$P/.venv/bin/python" - <<'PY'
from collections import Counter
import hashlib,json,os,subprocess,sys
from pathlib import Path
from routeset.observed_multitask import check_multitask_model_gate

p=Path('/home/wzy/dpvlm/route_set_v1');src=Path.cwd()
config_path=Path(os.environ['MULTITASK_LAUNCH_CONFIG']).resolve()
config=json.loads(config_path.read_text())
assert config['protocol']=='ordinary_multitask_free_offset_v1'
assert config['source_commit']==os.environ['CODE_COMMIT']==src.name
assert config['dataset']=='observation_multitask_prefix24_v1'
d=p/'data'/config['dataset'];out=p/'runs'/config['run_family'];cache=d/'qwen_cache'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(d/'snapshot_manifest.json')==config['snapshot_manifest_sha256']
assert sha(d/'observations.jsonl')==config['observations_sha256']
assert sha(d/'supervision.jsonl')==config['supervision_sha256']
gate=check_multitask_model_gate(d/'observations.jsonl',d/'supervision.jsonl')
assert gate is not None
rows=[json.loads(line) for line in (d/'observations.jsonl').read_text().splitlines()]
labels=[json.loads(line) for line in (d/'supervision.jsonl').read_text().splitlines()]
assert len(rows)==96 and Counter(row['split'] for row in rows)=={'TRAIN':48,'DEV_MODEL':48}
assert all(set(row)=={'id','parent_id','split','image','instruction'} for row in rows)
assert all(row['semantic_targets'] is None for row in labels)
assert len({row['parent_id'] for row in rows if row['split']=='TRAIN'})==12
assert len({row['parent_id'] for row in rows if row['split']=='DEV_MODEL'})==12
assert gate['collection_denominators']['successful_reference_routes']==72
assert not out.exists() and not cache.exists(),'Fresh pipeline only; inspect partial jobs and use exact recorded recovery commands deliberately.'
args=config['training_arguments']
assert (args['steps'],args['batch_size'],args['eval_every'],args['seed'])==(1500,32,250,0)
assert (args['endpoint_mode'],args['grounding_weight'],args['refinement_mode'])==('free_offset',0.,'none')
assert (args['sampling_mode'],args['metric_aggregation'])==('task_parent_language','task_parent')
out.mkdir(parents=True)
(out/'preregistered_config.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'pre_cache_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
sources=[Path(os.environ['MULTITASK_LAUNCHER']),config_path,
    src/'scripts/observation_cache_qwen.py',src/'scripts/train_observed_geometry.py',
    src/'scripts/train_observed_routes.py',src/'routeset/observed_geometry.py',
    src/'routeset/observed_route_head.py',src/'routeset/observed_multitask.py',
    src/'scripts/audit_multitask_layout_hashes.py',src/'routeset/multitask_fingerprints.py']
(out/'source_hashes.json').write_text(json.dumps({str(path):sha(path) for path in sources},indent=2)+'\n')
print('Verified exact 24-parent/96-language snapshot, current mechanical gate, and fresh outputs. Starting true Qwen cache then ordinary free-offset training.',flush=True)
def record(run_id,python,module,options,resume):
    command=[sys.executable,'-m','scripts.record_job','--output',str(out),'--run-id',run_id,'--resume-strategy',resume,'--',
             str(p/python),'-m',module]+options
    subprocess.run(command,check=True)
record('qwen_cache','.venv-qwen/bin/python','scripts.observation_cache_qwen',
    ['--model',str(p/'data/qwen3-vl-2b-instruct-89644892'),'--manifest',str(d/'observations.jsonl'),
     '--output',str(cache),'--device','cuda','--threads','1','--gpu-memory-fraction','0.35'],'none')
cache_config=json.loads((cache/'cache_config.json').read_text());cache_status=json.loads((cache/'status.json').read_text())
assert cache_config['revision']==cache_config['processor']==config['qwen_revision']
assert cache_config['manifest_sha256']==config['observations_sha256'] and cache_config['model_trainable_parameter_count']==0
assert cache_status['samples']==96 and cache_status['status']=='frozen_rgb_language_feature_extraction_complete'
options=['--observations',str(d/'observations.jsonl'),'--supervision',str(d/'supervision.jsonl'),
    '--cache-dir',str(cache),'--output',str(out/config['training_run_id']),
    '--multitask-snapshot-manifest',str(d/'snapshot_manifest.json')]
for name,value in args.items():options.extend(['--'+name.replace('_','-'),str(value)])
record(config['training_run_id'],'.venv/bin/python','scripts.train_observed_geometry',options,'flag')
print('Cache and free-offset training subprocesses both completed successfully; inspect actual best/last metrics before making any result claim.',flush=True)
PY
