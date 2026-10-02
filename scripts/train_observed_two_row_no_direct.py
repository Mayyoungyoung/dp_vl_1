"""One ordinary no-direct-Qwen control; historical model and loops unchanged."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts import train_observed_two_row_convergence64 as shared

PROTOCOL = 'ordinary_two_row_prefix76_no_direct_constant12000_v1'
MODEL_PROTOCOL = 'ordinary_geometry_no_direct_decoder_qwen_v1'
POLICY_FIELDS = ('no_direct_protocol', 'no_direct_model_protocol', 'no_direct_driver_sha256',
    'no_direct_model_sha256', 'no_direct_policy_sha256', 'no_direct_source_sha256',
    'no_direct_reference_sha256', 'no_direct_total_steps')
digest, write = shared.digest, shared.write


def validate_policy(p):
    expected = dict(protocol=PROTOCOL, model_protocol=MODEL_PROTOCOL, total_steps=12000,
        batch_size=32, candidates=4, seed=0, eval_every=250, lr=.0003,
        registered_train_parents=64, actual_train_parents=63, actual_train_inputs=189,
        fixed_dev_inputs=36, dev_selection_opportunities=48, new_training_observation_draws=384000,
        new_training_candidate_path_states=1536000, new_qwen_encodings=0,
        original_parameters=1231965, removed_parameters=549120, remaining_parameters=682845,
        data='data/observation_two_row_prefix76_v1', selection='configs/observed_two_row_prefix76_selection_v1.json',
        reference_run='runs/observed_two_row_prefix76_convergence_v1/peak_seed0',
        default_output='runs/observed_two_row_prefix76_no_direct_v1',
        lr_policy='unchanged constant LambdaLR=1', endpoint_clamping=False, geometry_frozen=False)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed no-direct control policy changed')
    for key in ('reference_last_sha256', 'reference_summary_sha256', 'reference_config_sha256',
                'reference_manifest_sha256', 'export_manifest_sha256'):
        value = p.get(key, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Frozen reference SHA required: ' + key)
    return p


def validate_resume(current, saved, total_steps):
    cfg = saved['config']
    if (any(k not in cfg or cfg[k] != current.get(k) for k in POLICY_FIELDS)
            or cfg.get('no_direct_protocol') != PROTOCOL or cfg.get('no_direct_model_protocol') != MODEL_PROTOCOL
            or cfg.get('steps') != total_steps or cfg.get('no_direct_total_steps') != total_steps):
        raise ValueError('No-direct resume protocol/source/total differs')
    if any(k.startswith('head.feature_encoder.') for k in saved['model']):
        raise ValueError('Historical direct-branch checkpoint cannot resume here')
    lr = current.get('lr', .0003)
    if (saved['scheduler'].get('base_lrs') != [lr] or saved['scheduler'].get('_last_lr') != [lr]
            or saved['scheduler'].get('last_epoch') != saved['step']
            or any(group['lr'] != lr for group in saved['optimizer']['param_groups'])):
        raise ValueError('Constant optimizer/scheduler state differs')


@contextmanager
def model_adapter(ordinary, metadata, expected_stream, total_steps=12000):
    """Scope only constructor, initial audit and declared step count, not loop."""
    import torch
    from routeset.observed_geometry_no_direct import ObservedGeometryNoDirectHead
    original_train, original_model = ordinary.base.train, ordinary.base.ObservedGeometryRouteHead
    original_audit = ordinary.base.new_stream_audit
    active_output = [None]
    starting_step = [0]
    hooks = []

    def initial_audit(model, sampler, rng):
        actual = original_audit(model, sampler, rng)
        proof = model.initialization_receipt
        if expected_stream:
            if proof['original_model_sha256'] != expected_stream['initial_model_sha256']:
                raise ValueError('Historical full construction initialization differs')
            for key in ('initial_torch_cpu_rng_sha256', 'initial_sampler_state_sha256'):
                if actual[key] != expected_stream[key]:
                    raise ValueError('Historical construction RNG/sampler differs: ' + key)
            if (proof['original_parameters'], proof['removed_parameters'], proof['remaining_parameters']) != (1231965, 549120, 682845):
                raise ValueError('Unexpected architecture/parameter reduction')
        if actual['initial_model_sha256'] != proof['shared_initialization_sha256']:
            raise ValueError('Live shared initialization differs from pre-removal tensors')
        if active_output[0] is not None:
            shared.write_once_equal(active_output[0]/'shared_initialization.json', proof)
        return actual

    def factory(*args, **kwargs):
        model = ObservedGeometryNoDirectHead(*args, **kwargs)
        names = {name for name, _ in model.named_parameters()}
        gradients = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                raise ValueError('All remaining modules must train')
            holder = []
            def record(grad, name=name, holder=holder):
                gradients[name] = dict(finite=bool(torch.isfinite(grad).all()),
                    nonzero=bool(torch.count_nonzero(grad)), l2=float(grad.detach().norm()))
                if holder: holder[0].remove()
                if set(gradients) == names and active_output[0] is not None:
                    write(active_output[0]/('first_backward_after_step%d.json' % starting_step[0]),
                        dict(start_step=starting_step[0], parameters=gradients,
                             all_remaining_parameters_received_gradient=True,
                             all_finite=all(v['finite'] for v in gradients.values())))
                return grad
            holder.append(parameter.register_hook(record)); hooks.extend(holder)
        return model

    def adapted(args):
        values = vars(args).copy()
        if values['steps'] != 1500:
            raise ValueError('Original recipe identity must remain1500')
        values.update(metadata, steps=total_steps, no_direct_total_steps=total_steps,
            no_direct_capacity_scope='549120 fewer parameters at production width; not a parameter-matched causal separation claim')
        active_output[0] = Path(args.output)
        if args.resume:
            saved = torch.load(active_output[0]/'last.pt', map_location='cpu', weights_only=False)
            validate_resume(values, saved, total_steps)
            starting_step[0] = saved['step']
        return original_train(SimpleNamespace(**values))

    ordinary.base.train, ordinary.base.ObservedGeometryRouteHead, ordinary.base.new_stream_audit = adapted, factory, initial_audit
    try:
        yield
    finally:
        for hook in hooks: hook.remove()
        ordinary.base.train, ordinary.base.ObservedGeometryRouteHead, ordinary.base.new_stream_audit = original_train, original_model, original_audit


def source_hashes(source):
    hashes = shared.source_hashes(source)
    for name in ('routeset/observed_geometry_no_direct.py', 'scripts/train_observed_two_row_no_direct.py',
                 'scripts/launch_observed_two_row_no_direct_v1.sh', 'tests/test_two_row_no_direct.py'):
        hashes[name] = digest(source/name)
    return hashes


def preflight(project, source, policy):
    from scripts.run_observed_two_row_scaling import validate_quality
    data, selection, quality, reference = (project/policy['data'], source/policy['selection'],
        project/policy['quality_audit'], project/policy['reference_run'])
    _, gate, rows = validate_quality(data, json.loads(selection.read_text()), quality)
    if digest(data/'export_manifest.json') != policy['export_manifest_sha256']:
        raise ValueError('Fixed export changed')
    for filename, key in (('last.pt', 'reference_last_sha256'), ('summary.json', 'reference_summary_sha256'),
                          ('config.json', 'reference_config_sha256')):
        if digest(reference/filename) != policy[key]:
            raise ValueError('Constant12000 original changed: ' + filename)
    ref_manifest = reference.parent/'convergence_manifest.json'
    if digest(ref_manifest) != policy['reference_manifest_sha256']:
        raise ValueError('Constant source manifest changed')
    sources = source_hashes(source)
    for name, value in json.loads(ref_manifest.read_text())['source_sha256'].items():
        if sources.get(name) != value:
            raise ValueError('Historical model/loop/checker source changed: ' + name)
    if sum(r['split']=='TRAIN' for r in rows)!=189 or sum(r['split']=='DEV_MODEL' for r in rows)!=36:
        raise ValueError('Exact existing TRAIN189/DEV36 required')
    return data, selection, quality, reference, gate, sources


def validate_finished(policy, saved, reference, summary, initialization):
    if (saved['step'] != 12000 or saved['config']['steps'] != 12000 or summary['last_step'] != 12000
            or saved['trajectory_exposures'] != 1536000 or summary['trajectory_exposures'] != 1536000
            or [r['step'] for r in saved['history']] != list(range(250,12001,250))):
        raise ValueError('Actual12000/1536000/48 evaluation budget differs')
    stream, expected = saved['sample_stream_audit'], policy['expected_sample_stream_audit']
    if reference['sample_stream_audit'] != expected or summary['sample_stream_audit'] != stream:
        raise ValueError('Reference/summary sample stream differs')
    for key in expected:
        if key != 'initial_model_sha256' and stream[key] != expected[key]:
            raise ValueError('Actual sampled sequence or construction RNG differs: ' + key)
    if (initialization['original_model_sha256'] != expected['initial_model_sha256']
            or initialization['shared_initialization_sha256'] != stream['initial_model_sha256']
            or not initialization['shared_tensor_bytes_exact'] or not initialization['rng_exact']):
        raise ValueError('Shared initialization proof differs')
    for key in ('rng','sampler_state','scheduler'):
        if shared.state_differences(saved[key], reference[key]):
            raise ValueError('Final RNG/sampler/constant schedule differs: '+key)
    for key in shared.CONFIG_FIELDS:
        if saved['config'].get(key) != reference['config'].get(key) or key not in saved['config']:
            raise ValueError('Historical training identity differs: '+key)
    if summary['parameters'] != 682845 or any(k.startswith('head.feature_encoder.') for k in saved['model']):
        raise ValueError('Removed branch or unexpected parameter budget')
    return dict(passed=True, actual_sample_stream=stream, reference_sample_stream=expected,
        shared_initialization_exact=True, full_initialization_equal=False, initial_forward_equal_claim=False,
        final_rng_and_sampler_exact=True, constant_scheduler_exact=True,
        original_parameters=1231965, removed_parameters=549120, remaining_parameters=682845,
        actual_training_observation_draws=384000, actual_training_candidate_path_states=1536000,
        dev_selection_opportunities=48, new_qwen_encodings=0,
        comparison_scope='Ordinary branch-removal/reduced-capacity control; indirect Qwen query remains in geometry context.')


def evaluate_fixed_last_train(model_folder, data_folder, output):
    """Same fixed-last diagnostic with an explicit no-direct checkpoint identity."""
    import torch
    from routeset.observed_geometry_no_direct import ObservedGeometryNoDirectHead
    from routeset.observed_route_head import load_observed_dataset
    from scripts.train_observed_geometry import load_geometry
    from scripts.train_observed_two_row import evaluate
    from scripts.evaluate_observed_two_row_online import head_options
    from scripts.audit_two_row_last_train import select_train
    from scripts.export_two_row_observations import verify_export
    model_folder,data_folder,output=map(Path,(model_folder,data_folder,output));started=time.perf_counter()
    torch.set_num_threads(1)
    _,gate=verify_export(data_folder)
    checkpoint_sha,export_sha=digest(model_folder/'last.pt'),digest(data_folder/'export_manifest.json')
    recovered=shared.recover_sealed_diagnostic(output,checkpoint_sha,export_sha)
    if recovered is not None:return recovered
    saved=torch.load(model_folder/'last.pt',map_location='cpu',weights_only=False);config=saved['config']
    if saved['step']!=12000 or config.get('no_direct_protocol')!=PROTOCOL:raise ValueError('Fixed no-direct last12000 required')
    model=ObservedGeometryNoDirectHead(**head_options(config)).eval();model.load_state_dict(saved['model'],strict=True)
    versions=tuple(p._version for p in model.parameters())
    data=load_observed_dataset(config['observations'],config['supervision'],config['cache_dir'],config['horizon'],config['pooling'])
    geometry=load_geometry(data,config['observations'],config['supervision'],config['pixel_stride'])
    if geometry['fingerprint']!=config['dataset_fingerprint']:raise ValueError('Last TRAIN fingerprint changed')
    ids=select_train(data,64)
    observed={r['id'] for r in map(json.loads,(data_folder/'observations.jsonl').read_text().splitlines()) if r['split']=='TRAIN'}
    if len(ids)!=189 or set(map(str,data['scene_ids'][ids]))!=observed:raise ValueError('All189 actual TRAIN inputs required')
    staging=output.with_name(output.name+'.staging');staging.mkdir()
    write(staging/'request_receipt.json',dict(checkpoint_sha256=checkpoint_sha,source_export_manifest_sha256=export_sha,
        reserved_forward_requests=189,reserved_complete_path_states=756,interruption_attempted_lower=0,interruption_attempted_upper=189,
        policy='Unsealed interruption is preserved; no automatic repeat forward.'))
    metrics=evaluate(model,data,geometry,ids,'cpu',staging,selection_metric='tip_unique_valid',
        evaluation_sources=(config['observations'],config['supervision']))
    if versions!=tuple(p._version for p in model.parameters()) or digest(model_folder/'last.pt')!=checkpoint_sha:
        raise ValueError('Diagnostic changed weights')
    verify_export(data_folder)
    receipt=dict(protocol=PROTOCOL,fixed_last_step=12000,checkpoint_sha256=checkpoint_sha,metrics=metrics,
        new_forward_requests=189,new_complete_path_states=756,new_qwen_encodings=0,new_dev_predictions=0,optimizer_updates=0,
        requested_train_inputs=192,actual_train_inputs=189,unavailable_train_inputs=3,
        prediction_sha256=digest(staging/'predictions.npz'),elapsed_seconds=time.perf_counter()-started,
        source_export_manifest_sha256=export_sha,initial_mechanical_gate=gate,execution_device='cpu',
        artifact_sha256={p.name:digest(p) for p in staging.iterdir() if p.is_file()})
    write(staging/'diagnostic_receipt.json',receipt);staging.rename(output);return receipt


def _train(args):
    import torch
    from routeset.observed_training_audit import tensor_state_digest
    from scripts import train_observed_two_row as ordinary
    project, source = Path(args.project).resolve(), Path(__file__).resolve().parents[1]
    policy_path = Path(args.policy).resolve(); policy = validate_policy(json.loads(policy_path.read_text()))
    output = project/policy['default_output']; model = output/'peak_seed0'
    data, selection, quality, reference, gate, sources = preflight(project, source, policy)
    metadata = dict(no_direct_protocol=PROTOCOL, no_direct_model_protocol=MODEL_PROTOCOL,
        no_direct_driver_sha256=digest(__file__), no_direct_model_sha256=sources['routeset/observed_geometry_no_direct.py'],
        no_direct_policy_sha256=digest(policy_path), no_direct_source_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        no_direct_reference_sha256=policy['reference_last_sha256'], no_direct_total_steps=12000)
    frozen = dict(protocol=PROTOCOL, policy=policy, metadata=metadata, source_sha256=sources)
    checkpoint = None
    if args.resume:
        if not (output/'no_direct_manifest.json').is_file(): raise ValueError('No original no-direct registration')
        checkpoint = torch.load(model/'last.pt', map_location='cpu', weights_only=False)
        validate_resume(dict(metadata,steps=12000,lr=policy['lr']),checkpoint,12000)
    elif model.exists() or (output/'no_direct_manifest.json').exists():
        raise FileExistsError('Fresh independent output required; explicit same-protocol resume only')
    output.mkdir(parents=True,exist_ok=True)
    shared.write_once_equal(output/'no_direct_manifest.json', frozen)
    write(output/'preflight_gate.json',gate)
    event = dict(resume=args.resume,start_step=checkpoint['step'] if checkpoint else 0,status='running',device=args.device)
    started = time.perf_counter()
    try:
        complete = checkpoint is not None and checkpoint['step']==12000
        event['optimization_skipped_for_metadata_recovery'] = complete
        if not complete:
            values = SimpleNamespace(data=str(data),selection=str(selection),quality_audit=str(quality),output=str(model),
                device=args.device,resume=args.resume,stop_after=None)
            with model_adapter(ordinary,metadata,policy['expected_sample_stream_audit']): ordinary.train(values)
        saved = torch.load(model/'last.pt',map_location='cpu',weights_only=False)
        shared.verify_ordinary_completion(model,saved['config'],ordinary.prediction_artifact_index)
        reference_saved = torch.load(reference/'last.pt',map_location='cpu',weights_only=False)
        result = json.loads((model/'summary.json').read_text())
        initialization = json.loads((model/'shared_initialization.json').read_text())
        paired = validate_finished(policy,saved,reference_saved,result,initialization)
        write(output/'paired_stream_receipt.json',paired)
        actual_names = set(saved['model'])
        if actual_names != set(initialization['initial_parameter_sha256']):
            raise ValueError('Remaining state dictionary changed its parameter names')
        updates = {name:dict(initial_sha256=initialization['initial_parameter_sha256'][name],
            final_sha256=tensor_state_digest({name:value}),
            changed=tensor_state_digest({name:value})!=initialization['initial_parameter_sha256'][name])
            for name,value in saved['model'].items()}
        gradients = sorted(model.glob('first_backward_after_step*.json'))
        if not gradients: raise ValueError('Actual first-backward gradient evidence missing')
        for path in gradients:
            receipt = json.loads(path.read_text())
            if (set(receipt['parameters']) != actual_names or not receipt['all_finite']
                    or not receipt['all_remaining_parameters_received_gradient']):
                raise ValueError('Remaining trainable parameter gradient audit incomplete')
        write(output/'remaining_parameter_updates.json',dict(parameters=updates,
            all_remaining_parameters_trainable=initialization['all_remaining_parameters_trainable'],
            changed_parameters=sum(v['changed'] for v in updates.values()),total_parameter_tensors=len(updates),
            gradient_artifacts={p.name:digest(p) for p in gradients}))
        diagnostic = evaluate_fixed_last_train(model,data,output/'fixed_last_train')
        preflight(project,source,policy)
        write(output/'no_direct_result_receipt.json',dict(paired,protocol=PROTOCOL,
            checkpoint_sha256={name:digest(model/name) for name in ('best.pt','last.pt')},
            summary_sha256=digest(model/'summary.json'),original_constant_preserved=True,
            prediction_artifacts=ordinary.prediction_artifact_index(model),fixed_last_train_diagnostic=diagnostic,
            shared_initialization_sha256=digest(model/'shared_initialization.json'),
            remaining_updates_sha256=digest(output/'remaining_parameter_updates.json'),
            actual_base_loop_elapsed_s=result['elapsed_s'],actual_base_loop_gpu_hours_reserved=result['gpu_hours_reserved'],
            cost_scope='Outer process includes initialization/evaluation/CPU diagnostic; nested base-loop cost is not additive.',
            scope='Single standard structural/capacity control, no core contribution or improvement promised.'))
        event.update(status='completed',end_step=saved['step'])
    except BaseException as exc:
        event.update(status='failed',exception=repr(exc),unsaved_optimizer_steps_upper_bound=250,
            unsaved_candidate_path_states_upper_bound=250*32*4,
            failure_budget_scope='Conservative one-checkpoint-interval bound; no automatic retry. Actual runtime remains charged.')
        raise
    finally:
        event['elapsed_seconds']=time.perf_counter()-started
        event['gpu_hours_reserved']=event['elapsed_seconds']/3600 if args.device.startswith('cuda') else 0.
        with (output/'runtime_events.jsonl').open('a',encoding='utf-8') as stream: stream.write(json.dumps(event,allow_nan=False)+'\n')


def train(args):
    policy=validate_policy(json.loads(Path(args.policy).read_text()))
    output=Path(args.project).resolve()/policy['default_output'];output.mkdir(parents=True,exist_ok=True)
    lock=output/'no_direct_pipeline.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try: _train(args)
    finally: lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='/home/wzy/dpvlm/route_set_v1')
    p.add_argument('--policy',default=str(Path(__file__).resolve().parents[1]/'configs/observed_two_row_no_direct_v1.json'))
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    train(p.parse_args())


if __name__=='__main__': main()
