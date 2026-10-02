"""Bounded12000 ordinary training, using the unchanged two-row/base loops.

The only scheduled optimization change is steps1500 ->12000 under the existing
constant LR. A separately launched finish stage requires exact first1500 proof.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

PROTOCOL='ordinary_two_row_prefix76_constant_lr_12000_v1'
STATE_FIELDS=('model','optimizer','scheduler','scaler','rng','sampler_state','sample_stream_audit',
              'step','trajectory_exposures','best','history')
CONFIG_FIELDS=('dataset_fingerprint','feature_dim','horizon','candidates','width','depth','point_width',
    'seed','lr','batch_size','event_scale','pooling','pixel_stride','endpoint_residual_bound','anchor_mode',
    'endpoint_mode','sampling_mode','grounding_weight','grounding_sigma','grounding_target','eval_every',
    'checkpoint_selection','sample_stream_audit','train_examples','dev_examples','train_parents','dev_parents',
    'two_row_driver_sha256','two_row_export_sha256','two_row_quality_audit_sha256','two_row_selection_sha256')
POLICY_FIELDS=('convergence_protocol','convergence_driver_sha256','convergence_policy_sha256',
    'convergence_reference_sha256','convergence_export_sha256','convergence_source_sha256')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_once_equal(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Existing sealed metadata differs: '+str(path))
    else:write(path,value)


def validate_policy(policy):
    expected=dict(protocol=PROTOCOL,registered_train_parents=64,actual_train_parents=63,actual_train_inputs=189,
        fixed_dev_inputs=36,original_steps=1500,stage1500_steps=1500,total_steps=12000,batch_size=32,
        candidates=4,seed=0,eval_every=250,lr=.0003,new_training_observation_draws=384000,
        new_training_candidate_path_states=1536000,new_qwen_encodings=0,dev_selection_opportunities=48)
    if any(policy.get(k)!=v for k,v in expected.items()):raise ValueError('Fixed prefix76 convergence policy changed')
    if policy.get('data')!='data/observation_two_row_prefix76_v1' or policy.get('selection')!='configs/observed_two_row_prefix76_selection_v1.json':
        raise ValueError('Only the existing prefix76 corpus is permitted')
    comparison=dict(run='runs/observed_two_row_prefix44_convergence_v1',actual_train_inputs=93,
        steps=6000,observation_draws=192000,candidate_path_states=768000,dev_selection_opportunities=24)
    if policy.get('comparison_reference')!=comparison:
        raise ValueError('Fixed32/6000 comparison identity or budget changed')
    for key in ('export_manifest_sha256','reference_last_sha256','reference_summary_sha256','reference_config_sha256','initialization_sha256'):
        value=policy.get(key,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Frozen source SHA required: '+key)
    return policy


def budget_receipt(policy):
    """Derive every reported budget from the frozen schedule, not copied literals."""
    p=validate_policy(policy);steps,batch,k=p['total_steps'],p['batch_size'],p['candidates']
    prefix=p['stage1500_steps'];remaining=steps-prefix;draws=steps*batch
    comparison=p['comparison_reference'];per_input=draws/p['actual_train_inputs']
    comparison_per_input=comparison['observation_draws']/comparison['actual_train_inputs']
    return dict(actual_total_steps=steps,actual_training_observation_draws=draws,
        actual_training_candidate_path_states=draws*k,
        original_reference_training_steps=p['original_steps'],
        original_reference_training_observation_draws=p['original_steps']*batch,
        original_reference_training_path_states=p['original_steps']*batch*k,
        new_run_reproduction_steps=prefix,new_run_reproduction_observation_draws=prefix*batch,
        new_run_reproduction_path_states=prefix*batch*k,
        new_run_postproof_steps=remaining,new_run_postproof_observation_draws=remaining*batch,
        new_run_postproof_path_states=remaining*batch*k,
        dev_selection_opportunities=steps//p['eval_every'],
        original_reference_dev_selection_opportunities=p['original_steps']//p['eval_every'],
        fixed_last_train_reserved_requests=p['actual_train_inputs'],
        fixed_last_train_reserved_path_states=p['actual_train_inputs']*k,
        per_actual_train_input_draws=per_input,
        comparison_reference=dict(comparison,per_actual_train_input_draws=comparison_per_input),
        per_input_draw_ratio_to_comparison=per_input/comparison_per_input,
        total_training_budget_ratio_to_comparison=draws/comparison['observation_draws'],
        comparison_scope='Approximate per-actual-input exposure alignment; unequal total training and DEV selection budgets.')


def validate_completed_budget(policy,result,checkpoint):
    budget=budget_receipt(policy);stream=result['sample_stream_audit']
    expected_history=list(range(policy['eval_every'],policy['total_steps']+1,policy['eval_every']))
    if (result['last_step']!=budget['actual_total_steps'] or checkpoint['step']!=budget['actual_total_steps']
            or checkpoint['config']['steps']!=budget['actual_total_steps']
            or stream['batches']!=budget['actual_total_steps']
            or stream['observation_draws']!=budget['actual_training_observation_draws']
            or result['trajectory_exposures']!=budget['actual_training_candidate_path_states']
            or checkpoint['trajectory_exposures']!=budget['actual_training_candidate_path_states']
            or stream['initial_model_sha256']!=policy['initialization_sha256']
            or [row['step'] for row in checkpoint['history']]!=expected_history):
        raise ValueError('Actual convergence budget, initialization or DEV selection schedule differs')
    return budget


def state_differences(left,right,path=''):
    """Exact nested value comparison; checkpoint-container bytes may differ."""
    if isinstance(left,dict) and isinstance(right,dict):
        if set(left)!=set(right):return [path+':keys']
        return [item for key in left for item in state_differences(left[key],right[key],path+'/'+str(key))]
    if isinstance(left,(list,tuple)) and isinstance(right,type(left)):
        if len(left)!=len(right):return [path+':length']
        return [item for i,(a,b) in enumerate(zip(left,right)) for item in state_differences(a,b,path+'/'+str(i))]
    if hasattr(left,'detach') or hasattr(right,'detach'):
        if not (hasattr(left,'detach') and hasattr(right,'detach')):return [path+':tensor_type']
        if left.dtype!=right.dtype or left.shape!=right.shape:return [path+':tensor_shape_dtype']
        import torch
        return [] if torch.equal(left.detach().cpu(),right.detach().cpu()) else [path+':tensor_values']
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return [] if (isinstance(left,np.ndarray) and isinstance(right,np.ndarray) and left.dtype==right.dtype
            and left.shape==right.shape and np.array_equal(left,right)) else [path+':array_values']
    return [] if type(left)==type(right) and left==right else [path+':value']


def compare_training_state(reference,current):
    fields={key:([key+':missing_required_key'] if key not in reference or key not in current
        else state_differences(reference[key],current[key],key)) for key in STATE_FIELDS}
    a,b=reference.get('config',{}),current.get('config',{})
    fields['config_training_identity']=[key for key in CONFIG_FIELDS if key not in a or key not in b or state_differences(a[key],b[key])]
    return dict(protocol='exact_constant_lr_prefix_reproduction_v1',passed=all(not v for v in fields.values()),
        differing_fields={k:v[:100] for k,v in fields.items() if v},compared_fields=list(fields),
        ignored=['checkpoint container hash','top-level elapsed_s','config steps/output/source path/stop_after/resume and timing metadata'],
        tolerance=0)


def validate_resume_policy(current,saved,total_steps=12000):
    if any(current.get(key)!=saved.get(key) or key not in saved for key in POLICY_FIELDS):
        raise ValueError('Convergence resume policy/source/reference differs')
    if saved.get('steps')!=total_steps or current.get('steps')!=total_steps:raise ValueError('Convergence resume total-step policy differs')
    if (saved.get('convergence_selection_budget_steps')!=1500 or saved.get('convergence_actual_total_steps')!=total_steps
            or saved.get('convergence_lr_policy')!='unchanged constant LambdaLR=1'):
        raise ValueError('Saved explicit convergence schedule metadata differs')


@contextmanager
def schedule_adapter(ordinary,metadata,total_steps=12000):
    original=ordinary.base.train
    def adapted(args):
        values=vars(args).copy()
        if values['steps']!=1500:raise ValueError('The original ordinary recipe must remain1500')
        values.update(metadata,steps=total_steps,convergence_selection_budget_steps=1500,
            convergence_actual_total_steps=total_steps,convergence_lr_policy='unchanged constant LambdaLR=1')
        if args.resume:
            import torch
            saved=torch.load(Path(args.output)/'last.pt',map_location='cpu',weights_only=False)
            validate_resume_policy(values,saved['config'],total_steps);del saved
        return original(SimpleNamespace(**values))
    ordinary.base.train=adapted
    try:yield
    finally:ordinary.base.train=original


def source_hashes(source):
    names=('scripts/train_observed_two_row_convergence64.py','scripts/train_observed_two_row.py',
        'scripts/train_observed_geometry.py','scripts/train_observed_routes.py','scripts/export_two_row_observations.py',
        'scripts/run_observed_two_row_scaling.py','scripts/evaluate_observed_two_row.py','scripts/evaluate_observed_obstacles.py',
        'routeset/observed_geometry.py','routeset/observed_route_head.py','routeset/train_v2.py',
        'routeset/observed_training_audit.py','routeset/observed_grounding_targets.py','routeset/observed_multitask.py',
        'scripts/audit_two_row_last_train.py','scripts/evaluate_observed_two_row_online.py')
    return {name:digest(source/name) for name in names}


def metadata_for(policy,policy_path,source):
    hashes=source_hashes(source)
    return dict(convergence_protocol=PROTOCOL,convergence_driver_sha256=digest(__file__),
        convergence_policy_sha256=digest(policy_path),convergence_reference_sha256=policy['reference_last_sha256'],
        convergence_export_sha256=policy['export_manifest_sha256'],
        convergence_source_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()),hashes


def validate_stage(stage,resume,output,checkpoint_step=None):
    model=Path(output)/'peak_seed0'
    if stage=='stage1500':
        if not resume and model.exists():raise FileExistsError('Fresh reproduction required; preserve original and interrupted evidence')
        if resume and (checkpoint_step is None or not 0<checkpoint_step<=1500):raise ValueError('Resume stage1500 only through its registered boundary')
    elif stage=='finish':
        if checkpoint_step is None or not 1500<=checkpoint_step<=12000:raise ValueError('Finish requires a12000-policy checkpoint at or after1500')
        if not resume and checkpoint_step!=1500:raise ValueError('Interrupted finish requires explicit --resume')
    else:raise ValueError('Only stage1500 or finish is registered')


def preflight(project,source,policy):
    from scripts.run_observed_two_row_scaling import validate_quality
    data=project/policy['data'];selection=source/policy['selection'];quality=project/policy['quality_audit'];reference=project/policy['reference_run']
    manifest,gate,rows=validate_quality(data,json.loads(selection.read_text()),quality)
    if digest(data/'export_manifest.json')!=policy['export_manifest_sha256']:raise ValueError('Fixed prefix76 export changed')
    for name,key in (('last.pt','reference_last_sha256'),('summary.json','reference_summary_sha256'),('config.json','reference_config_sha256')):
        if digest(reference/name)!=policy[key]:raise ValueError('Original1500 evidence changed: '+name)
    if sum(row['split']=='TRAIN' for row in rows)!=189 or sum(row['split']=='DEV_MODEL' for row in rows)!=36:
        raise ValueError('Original actual TRAIN189/DEV36 observations required')
    return data,selection,quality,reference,gate


def snapshot1500(output):
    output=Path(output);model=output/'peak_seed0';snapshot=output/'stage1500_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ('last.pt','best.pt','config.json','history.json','status.json','source_hashes.json'):
        if (snapshot/name).exists():
            if digest(snapshot/name)!=digest(model/name):raise ValueError('Never replace a different stage1500 snapshot')
        else:shutil.copyfile(model/name,snapshot/name)
    index={name.name:dict(sha256=digest(name),bytes=name.stat().st_size) for name in snapshot.iterdir() if name.name!='artifact_index.json'}
    write_once_equal(snapshot/'artifact_index.json',index);return index


def verify_stage1500_gate(output,policy,metadata):
    output=Path(output);audit=json.loads((output/'stage1500_audit.json').read_text())
    if (not audit['passed'] or audit['reference_sha256']!=policy['reference_last_sha256']
            or audit['metadata']!=metadata or audit['step']!=1500):raise ValueError('Passing unchanged exact stage1500 proof required')
    snapshot=output/'stage1500_snapshot';index=json.loads((snapshot/'artifact_index.json').read_text())
    for name,value in index.items():
        if digest(snapshot/name)!=value['sha256']:raise ValueError('Stage1500 snapshot changed')
    if digest(snapshot/'last.pt')!=audit['reproduced_sha256']:raise ValueError('Audited stage1500 checkpoint changed')
    return audit


def recover_sealed_diagnostic(output,checkpoint_sha,export_sha):
    """Return a completed receipt without another prediction call; fail closed otherwise."""
    output=Path(output);staging=output.with_name(output.name+'.staging')
    existing=output if output.exists() else staging if staging.exists() else None
    if existing is None:return None
    receipt_path=existing/'diagnostic_receipt.json'
    if not receipt_path.exists():
        raise ValueError('Unsealed diagnostic remains; preserve all files and its189-request budget upper bound. No automatic new forward.')
    receipt=json.loads(receipt_path.read_text())
    if receipt['checkpoint_sha256']!=checkpoint_sha or receipt['source_export_manifest_sha256']!=export_sha:
        raise ValueError('Sealed diagnostic source differs')
    for name,value in receipt['artifact_sha256'].items():
        if Path(name).name!=name or digest(existing/name)!=value:raise ValueError('Sealed diagnostic pool SHA changed')
    if existing==staging:staging.rename(output)
    return receipt


def verify_ordinary_completion(model,config,index_builder):
    """Require the existing driver's sealed pools; never replay final inference."""
    model=Path(model);path=model/'two_row_driver_receipt.json'
    if not path.is_file():
        raise ValueError('Unsealed ordinary completion; preserve final checkpoint/pools for manual sealing. No automatic final forward.')
    receipt=json.loads(path.read_text())
    for key,config_key in (('protocol','two_row_driver_protocol'),('source_sha256','two_row_driver_sha256'),
            ('export_sha256','two_row_export_sha256'),('quality_audit_sha256','two_row_quality_audit_sha256')):
        if key not in receipt or config_key not in config or receipt[key]!=config[config_key]:
            raise ValueError('Ordinary completion identity differs: '+key)
    if (receipt.get('actual_training_summary_sha256')!=digest(model/'summary.json')
            or receipt.get('training_candidate_path_states')!=1536000):
        raise ValueError('Ordinary completion summary/budget differs')
    actual=index_builder(model)
    if not actual or receipt.get('prediction_artifacts')!=actual:
        raise ValueError('Ordinary completion prediction pool seal differs')
    return receipt


def evaluate_fixed_last_train(model_folder,data_folder,output,expected_step=12000):
    """One CPU prediction pool for all actual TRAIN inputs; no reselection."""
    import torch
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_route_head import load_observed_dataset
    from scripts.train_observed_geometry import load_geometry
    from scripts.train_observed_two_row import evaluate
    from scripts.evaluate_observed_two_row_online import head_options
    from scripts.audit_two_row_last_train import select_train
    from scripts.export_two_row_observations import verify_export
    model_folder,data_folder,output=map(Path,(model_folder,data_folder,output));started=time.perf_counter()
    torch.set_num_threads(1)
    manifest,gate=verify_export(data_folder)
    source_sha=digest(model_folder/'last.pt');export_sha=digest(data_folder/'export_manifest.json')
    recovered=recover_sealed_diagnostic(output,source_sha,export_sha)
    if recovered is not None:return recovered
    saved=torch.load(model_folder/'last.pt',map_location='cpu',weights_only=False);config=saved['config']
    if saved['step']!=expected_step or config['convergence_protocol']!=PROTOCOL:raise ValueError('Fixed convergence-last checkpoint required')
    model=ObservedGeometryRouteHead(**head_options(config)).eval();model.load_state_dict(saved['model'],strict=True)
    versions=tuple(p._version for p in model.parameters())
    data=load_observed_dataset(config['observations'],config['supervision'],config['cache_dir'],config['horizon'],config['pooling'])
    geometry=load_geometry(data,config['observations'],config['supervision'],config['pixel_stride'])
    if geometry['fingerprint']!=config['dataset_fingerprint']:raise ValueError('Last TRAIN dataset fingerprint changed')
    ids=select_train(data,64)
    observed={r['id'] for r in map(json.loads,(data_folder/'observations.jsonl').read_text().splitlines()) if r['split']=='TRAIN'}
    if len(ids)!=189 or set(map(str,data['scene_ids'][ids]))!=observed:raise ValueError('All189 actual TRAIN observations required')
    staging=output.with_name(output.name+'.staging');staging.mkdir()
    write(staging/'request_receipt.json',dict(checkpoint_sha256=source_sha,source_export_manifest_sha256=export_sha,
        reserved_forward_requests=189,reserved_complete_path_states=756,interruption_attempted_lower=0,interruption_attempted_upper=189,
        policy='Unsealed interruption never triggers automatic replay; preserve staging and full reserved diagnostic budget.'))
    metrics=evaluate(model,data,geometry,ids,'cpu',staging,selection_metric='tip_unique_valid',
        evaluation_sources=(config['observations'],config['supervision']))
    if versions!=tuple(p._version for p in model.parameters()) or digest(model_folder/'last.pt')!=source_sha:raise ValueError('Diagnostic changed weights')
    verify_export(data_folder)
    receipt=dict(fixed_last_step=expected_step,checkpoint_sha256=source_sha,metrics=metrics,new_forward_requests=189,
        new_complete_path_states=756,new_qwen_encodings=0,new_dev_predictions=0,optimizer_updates=0,
        requested_train_inputs=192,actual_train_inputs=189,unavailable_train_inputs=3,
        prediction_sha256=digest(staging/'predictions.npz'),elapsed_seconds=time.perf_counter()-started,
        source_export_manifest_sha256=export_sha,initial_mechanical_gate=gate,
        artifact_sha256={p.name:digest(p) for p in staging.iterdir() if p.is_file()},
        execution_device='cpu',scope='Fixed-last TRAIN fit diagnostic; best selection and original pools remain unchanged.')
    write(staging/'diagnostic_receipt.json',receipt);staging.rename(output);return receipt


def train(args):
    import torch
    from scripts import train_observed_two_row as ordinary
    project,source=Path(args.project).resolve(),Path(__file__).resolve().parents[1]
    policy_path=Path(args.policy).resolve();policy=validate_policy(json.loads(policy_path.read_text()))
    output=Path(args.output).resolve() if args.output else project/policy['default_output']
    data,selection,quality,reference,gate=preflight(project,source,policy)
    metadata,hashes=metadata_for(policy,policy_path,source)
    model=output/'peak_seed0';current=None
    if (model/'last.pt').exists():current=torch.load(model/'last.pt',map_location='cpu',weights_only=False)
    validate_stage(args.stage,args.resume,output,None if current is None else current['step'])
    if current is not None:validate_resume_policy(dict(metadata,steps=12000),current['config'])
    if args.stage=='finish':
        proof=verify_stage1500_gate(output,policy,metadata)
        if current['step']==1500 and digest(model/'last.pt')!=proof['reproduced_sha256']:
            raise ValueError('Finish did not start from audited1500 state')
    manifest=output/'convergence_manifest.json'
    frozen=dict(protocol=PROTOCOL,policy=policy,policy_sha256=digest(policy_path),metadata=metadata,
        source_sha256=hashes,data=str(data),selection=str(selection),quality=str(quality),reference_run=str(reference))
    if manifest.exists():
        if json.loads(manifest.read_text())!=frozen:raise ValueError('Convergence manifest/source changed')
    else:
        if args.stage!='stage1500' or args.resume:raise ValueError('Fresh convergence registration required')
        output.mkdir(parents=True,exist_ok=True);write(manifest,frozen)
    write(output/(args.stage+'_preflight_gate.json'),gate)
    started=time.perf_counter();event=dict(stage=args.stage,resume=args.resume,device=args.device,
        start_step=0 if current is None else current['step'],status='running')
    already_trained=current is not None and ((args.stage=='stage1500' and current['step']==1500)
        or current['step']==12000)
    event['optimization_skipped_for_metadata_recovery']=already_trained
    del current
    try:
        values=SimpleNamespace(data=str(data),selection=str(selection),quality_audit=str(quality),output=str(model),
            device=args.device,resume=args.resume or args.stage=='finish',stop_after=1500 if args.stage=='stage1500' else None)
        if not already_trained:
            with schedule_adapter(ordinary,metadata):ordinary.train(values)
        checkpoint=torch.load(model/'last.pt',map_location='cpu',weights_only=False)
        if args.stage=='stage1500':
            original=torch.load(reference/'last.pt',map_location='cpu',weights_only=False)
            audit=compare_training_state(original,checkpoint)
            audit.update(step=checkpoint['step'],reference_sha256=digest(reference/'last.pt'),reproduced_sha256=digest(model/'last.pt'),metadata=metadata)
            audit['passed']=bool(audit['passed'] and checkpoint['step']==1500
                and checkpoint['sample_stream_audit']['initial_model_sha256']==policy['initialization_sha256'])
            snapshot1500(output);write_once_equal(output/'stage1500_audit.json',audit)
            if not audit['passed']:raise ValueError('Exact1500 reproduction failed; no finish or tolerance fallback')
        else:
            verify_ordinary_completion(model,checkpoint['config'],ordinary.prediction_artifact_index)
            result=json.loads((model/'summary.json').read_text())
            budget=validate_completed_budget(policy,result,checkpoint)
            diagnostic=evaluate_fixed_last_train(model,data,output/'fixed_last_train')
            write(output/'fixed_last_train_receipt.json',diagnostic)
            write(output/'convergence_result_receipt.json',dict(budget,protocol=PROTOCOL,
                new_qwen_encodings=0,shared_frozen_cache=str(data/'qwen_cache'),
                original_reference_preserved=digest(reference/'last.pt')==policy['reference_last_sha256'],
                stage1500_audit_sha256=digest(output/'stage1500_audit.json'),summary_sha256=digest(model/'summary.json'),
                checkpoint_sha256={name:digest(model/name) for name in ('best.pt','last.pt')},
                prediction_artifacts=ordinary.prediction_artifact_index(model),
                fixed_last_train_diagnostic=diagnostic,fixed_last_train_receipt_sha256=digest(output/'fixed_last_train_receipt.json'),
                actual_base_loop_elapsed_s=result['elapsed_s'],actual_base_loop_gpu_hours_reserved=result['gpu_hours_reserved'],
                total_driver_cost_source='runtime_events.jsonl plus recorded stage process status; base-loop cost is nested, not added again',
                scope='More fully trained ordinary baseline. The original1500 selection is data identity, not the actual12000 budget.'))
        event.update(status='completed',end_step=checkpoint['step'])
    except BaseException as exc:
        event.update(status='failed',exception=repr(exc));raise
    finally:
        event['elapsed_seconds']=time.perf_counter()-started
        event['gpu_hours_reserved']=event['elapsed_seconds']/3600 if args.device.startswith('cuda') else 0.
        with (output/'runtime_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='/home/wzy/dpvlm/route_set_v1')
    parser.add_argument('--policy',default=str(Path(__file__).resolve().parents[1]/'configs/observed_two_row_convergence64_v1.json'))
    parser.add_argument('--output');parser.add_argument('--stage',choices=('stage1500','finish'),required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--device',default='cuda')
    train(parser.parse_args())


if __name__=='__main__':main()
