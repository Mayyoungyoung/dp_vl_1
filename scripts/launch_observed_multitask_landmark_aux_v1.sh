#!/usr/bin/env bash
# A single ordinary auxiliary-loss control; root reviews and starts the frozen copy.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION=aef39831c4429474c27acbc69152191cb345b454
SRC="$P/research_v2/releases/$REVISION"
CONFIG="${1:?Reviewed immutable pair configuration path required}"
cd "$SRC"
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CODE_COMMIT="$REVISION"
export LANDMARK_CONFIG="$CONFIG" LANDMARK_LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
"$P/.venv/bin/python" - <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
from routeset.observed_multitask import check_multitask_model_gate
p=Path('/home/wzy/dpvlm/route_set_v1');src=Path.cwd()
config_path=Path(os.environ['LANDMARK_CONFIG']).resolve();config=json.loads(config_path.read_text())
assert config['protocol']=='ordinary_multitask_event_supported_aux_pair_v1'
assert src.name==os.environ['CODE_COMMIT']=='aef39831c4429474c27acbc69152191cb345b454'
assert config['dataset']=='observation_multitask_prefix108_v1'
assert config['proposed_output']=='observed_multitask_landmark_aux_v1/event_supported_seed0'
d=p/'data'/config['dataset'];out=p/'runs/observed_multitask_landmark_aux_v1';run=out/'event_supported_seed0'
control=Path(config['ordinary_control']);summary=json.loads((control/'summary.json').read_text())
proof=p/'runs/observed_multitask_landmark_aux_preflight_v1/actual_control'
receipt=json.loads((proof/'summary.json').read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert receipt['status']=='passed' and receipt['initial_forward_bit_equal'] and receipt['sampler_final_state_equals_completed_control']
assert receipt['code_commit']==src.name and receipt['ordinary_run']==str(control)
assert receipt['ordinary_checkpoint_sha256']==summary['last_checkpoint_sha256']==sha(control/'last.pt')
assert receipt['actual_target_helper_sha256']==sha(src/'routeset/observed_grounding_targets.py')
assert receipt['actual_new_trainer_sha256']==sha(src/'scripts/train_observed_geometry.py')
assert receipt['target_selection_sha256']==sha(proof/'grounding_target_selection.json')
assert sha(d/'snapshot_manifest.json')==config['snapshot_manifest_sha256']
assert sha(d/'observations.jsonl')==config['observations_sha256'] and sha(d/'supervision.jsonl')==config['supervision_sha256']
gate=check_multitask_model_gate(d/'observations.jsonl',d/'supervision.jsonl')
assert gate is not None and gate['collection_denominators']['successful_reference_routes']==321
cache=json.loads((d/'qwen_cache/cache_config.json').read_text())
assert cache['revision']==cache['processor']=='89644892e4d85e24eaac8bacfd4f463576704203'
assert cache['manifest_sha256']==config['observations_sha256'] and cache['model_trainable_parameter_count']==0
args=config['training_arguments']
assert (args['steps'],args['batch_size'],args['candidates'],args['seed'],args['eval_every'])==(1500,32,4,0,250)
assert (args['endpoint_mode'],args['refinement_mode'],args['grounding_target'])==('free_offset','none','event_supported')
assert (args['grounding_weight'],args['grounding_sigma'])==(.02,.025)
control_config=json.loads((control/'config.json').read_text())
for name,value in args.items():
    if name not in ('grounding_target','grounding_weight'):
        key='checkpoint_selection' if name=='selection_metric' else name
        assert control_config[key]==value,'Unexpected difference from ordinary training: '+name
assert (receipt['positive_train_observations'],receipt['unique_positive_reference_routes'])==(381,285)
assert not out.exists(),'Fresh output only; inspect any partial job and its explicit resume record.'
out.mkdir(parents=True)
(out/'preregistered_pair.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'preflight_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(out/'current_model_use_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
sources=[Path(os.environ['LANDMARK_LAUNCHER']),config_path,src/'scripts/train_observed_geometry.py',
    src/'routeset/observed_grounding_targets.py',src/'routeset/observed_geometry.py',src/'routeset/observed_route_head.py',
    src/'routeset/observed_multitask.py',src/'routeset/train_v2.py',proof/'summary.json',proof/'grounding_target_selection.json']
(out/'source_hashes.json').write_text(json.dumps({str(path):sha(path) for path in sources},indent=2)+'\n')
options=['--observations',str(d/'observations.jsonl'),'--supervision',str(d/'supervision.jsonl'),
    '--cache-dir',str(d/'qwen_cache'),'--output',str(run),'--multitask-snapshot-manifest',str(d/'snapshot_manifest.json')]
for name,value in args.items():options.extend(['--'+name.replace('_','-'),str(value)])
print('Verified same-data/source-compatible initialization and sampler replay; starting one fixed1500x32 auxiliary-loss control.',flush=True)
subprocess.run([sys.executable,'-m','scripts.record_job','--output',str(out),'--run-id','event_supported_seed0',
    '--resume-strategy','flag','--',sys.executable,'-m','scripts.train_observed_geometry']+options,check=True)
actual=json.loads((run/'config.json').read_text())
assert actual['grounding_target_fingerprint']==receipt['grounding_target_fingerprint']
print('Completed auxiliary control; compare actual best and last with unchanged ordinary prefix108 metrics.',flush=True)
PY
