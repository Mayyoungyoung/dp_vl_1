"""One fresh, fixed-budget cosine control; historical model/loop files stay intact."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from scripts import train_observed_two_row_convergence64 as shared

PROTOCOL = 'ordinary_two_row_prefix76_cosine_12000_v1'
SCHEDULE = 'single_cycle_cosine_no_warmup_no_restart_no_floor_v1'
POLICY_FIELDS = ('cosine_protocol', 'cosine_driver_sha256', 'cosine_policy_sha256',
    'cosine_source_sha256', 'cosine_reference_sha256', 'cosine_schedule', 'cosine_total_steps')
digest, write = shared.digest, shared.write


def factor(completed_steps, total_steps):
    if total_steps <= 0 or not 0 <= completed_steps <= total_steps:
        raise ValueError('Cosine schedule step outside its fixed cycle')
    return (1. + math.cos(math.pi * completed_steps / total_steps)) / 2.


def validate_lr_state(state, total_steps, expected_step=None):
    step = state.get('last_epoch')
    if (state.get('cosine_schedule') != SCHEDULE or state.get('cosine_total_steps') != total_steps
            or not isinstance(step, int) or not 0 <= step <= total_steps
            or state.get('_cosine_initializing') is not False or state.get('_step_count') != step + 1
            or (expected_step is not None and step != expected_step)):
        raise ValueError('Scheduler protocol/step differs')
    used, bases = state.get('used_lrs'), state.get('base_lrs')
    if not bases or not isinstance(used, list) or len(used) != step:
        raise ValueError('Actual per-step LR trace missing')
    for i, values in enumerate(used):
        if values != [base * factor(i, total_steps) for base in bases]:
            raise ValueError('Actual optimizer LR trace differs at step ' + str(i + 1))
    if state.get('_last_lr') != [base * factor(step, total_steps) for base in bases]:
        raise ValueError('Next optimizer LR differs')
    return used


def make_scheduler(original_class, optimizer, total_steps, kind='cosine'):
    """Class methods, not instance closures, keep state_dict fully serializable."""
    if kind == 'constant_test_control':
        return original_class(optimizer, lambda _: 1.)
    if kind != 'cosine':
        raise ValueError('Unknown test schedule')

    class AuditedCosine(original_class):
        def __init__(self):
            self.cosine_schedule = SCHEDULE
            self.cosine_total_steps = total_steps
            self.used_lrs = []
            self._cosine_initializing = True
            super().__init__(optimizer, lambda step: factor(step, total_steps))
            self._cosine_initializing = False

        def step(self, epoch=None):
            if epoch is not None:
                raise ValueError('No explicit epoch jumps are permitted')
            values = [group['lr'] for group in self.optimizer.param_groups]
            if not self._cosine_initializing:
                if self.last_epoch >= self.cosine_total_steps:
                    raise ValueError('No schedule extension after fixed total steps')
                expected = [base * factor(self.last_epoch, self.cosine_total_steps) for base in self.base_lrs]
                if values != expected:
                    raise ValueError('Optimizer LR changed outside the fixed scheduler')
            super().step()
            if not self._cosine_initializing:
                self.used_lrs.append(values)

        def load_state_dict(self, state):
            validate_lr_state(state, self.cosine_total_steps)
            if state['base_lrs'] != self.base_lrs:
                raise ValueError('Scheduler base LR differs from declared optimizer initialization')
            if [group['lr'] for group in self.optimizer.param_groups] != state['_last_lr']:
                raise ValueError('Restored optimizer and scheduler LR disagree')
            super().load_state_dict(state)

    return AuditedCosine()


def validate_policy(p):
    expected = dict(protocol=PROTOCOL, schedule=SCHEDULE, total_steps=12000, batch_size=32,
        candidates=4, seed=0, eval_every=250, lr=.0003, registered_train_parents=64,
        actual_train_parents=63, actual_train_inputs=189, fixed_dev_inputs=36,
        dev_selection_opportunities=48, new_training_observation_draws=384000,
        new_training_candidate_path_states=1536000, new_qwen_encodings=0,
        warmup_steps=0, restart_count=0, lr_floor=0.,
        data='data/observation_two_row_prefix76_v1', selection='configs/observed_two_row_prefix76_selection_v1.json',
        default_output='runs/observed_two_row_prefix76_cosine_v1',
        reference_run='runs/observed_two_row_prefix76_convergence_v1/peak_seed0')
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Fixed cosine control policy changed')
    for k in ('reference_last_sha256', 'reference_summary_sha256', 'reference_config_sha256',
              'reference_manifest_sha256', 'export_manifest_sha256'):
        value = p.get(k, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Frozen reference SHA required: ' + k)
    return p


def validate_resume(current, checkpoint, total_steps):
    saved = checkpoint['config']
    if (any(k not in saved or saved[k] != current.get(k) for k in POLICY_FIELDS)
            or saved.get('cosine_protocol') != PROTOCOL or saved.get('cosine_schedule') != SCHEDULE
            or saved.get('steps') != total_steps or saved.get('cosine_total_steps') != total_steps):
        raise ValueError('Cosine resume protocol/source/total differs; constant checkpoints are not resumable here')
    validate_lr_state(checkpoint['scheduler'], total_steps, checkpoint['step'])
    if checkpoint['scheduler']['base_lrs'] != [current.get('lr', .0003)] or saved.get('lr') != current.get('lr', .0003):
        raise ValueError('Saved scheduler base LR differs from fixed initial LR')


@contextmanager
def schedule_adapter(ordinary, metadata, expected_stream, total_steps=12000, kind='cosine'):
    import torch
    original_train = ordinary.base.train
    original_scheduler = torch.optim.lr_scheduler.LambdaLR
    original_audit = ordinary.base.new_stream_audit

    def initial_audit(*args, **kwargs):
        actual = original_audit(*args, **kwargs)
        for key in ('initial_model_sha256', 'initial_torch_cpu_rng_sha256', 'initial_sampler_state_sha256'):
            if expected_stream and actual[key] != expected_stream[key]:
                raise ValueError('Initial tensor/RNG/sampler differs from constant control: ' + key)
        return actual

    def scheduler(optimizer, ignored_lambda):
        return make_scheduler(original_scheduler, optimizer, total_steps, kind)

    def adapted(args):
        values = vars(args).copy()
        if values['steps'] != 1500:
            raise ValueError('Original recipe identity must remain1500')
        values.update(metadata, steps=total_steps, cosine_total_steps=total_steps, cosine_schedule=SCHEDULE)
        if args.resume:
            saved = torch.load(Path(args.output)/'last.pt', map_location='cpu', weights_only=False)
            validate_resume(values, saved, total_steps)
        return original_train(SimpleNamespace(**values))

    ordinary.base.train, ordinary.base.new_stream_audit = adapted, initial_audit
    torch.optim.lr_scheduler.LambdaLR = scheduler
    try:
        yield
    finally:
        ordinary.base.train, ordinary.base.new_stream_audit = original_train, original_audit
        torch.optim.lr_scheduler.LambdaLR = original_scheduler


def source_hashes(source):
    hashes = shared.source_hashes(source)
    for name in ('scripts/train_observed_two_row_cosine.py', 'scripts/launch_observed_two_row_cosine_v1.sh',
                 'tests/test_two_row_cosine.py'):
        hashes[name] = digest(source/name)
    return hashes


def preflight(project, source, policy):
    from scripts.run_observed_two_row_scaling import validate_quality
    data, selection, quality, reference = (project/policy['data'], source/policy['selection'],
        project/policy['quality_audit'], project/policy['reference_run'])
    manifest, gate, rows = validate_quality(data, json.loads(selection.read_text()), quality)
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


def validate_finished(policy, saved, reference, summary):
    if (saved['step'] != 12000 or saved['config']['steps'] != 12000 or summary['last_step'] != 12000
            or saved['trajectory_exposures'] != 1536000 or summary['trajectory_exposures'] != 1536000
            or [x['step'] for x in saved['history']] != list(range(250,12001,250))):
        raise ValueError('Actual12000/1536000/48 evaluation budget differs')
    stream = saved['sample_stream_audit']
    if stream != policy['expected_sample_stream_audit'] or stream != reference['sample_stream_audit'] or summary['sample_stream_audit'] != stream:
        raise ValueError('Actual initialization or sampled observation sequence differs')
    differences = {key:shared.state_differences(saved[key], reference[key]) for key in ('rng','sampler_state')}
    if any(differences.values()):
        raise ValueError('Final RNG/sampler differs from constant control')
    for key in shared.CONFIG_FIELDS:
        if key not in saved['config'] or key not in reference['config'] or saved['config'][key] != reference['config'][key]:
            raise ValueError('Training identity differs from constant control: '+key)
    used = validate_lr_state(saved['scheduler'], 12000, 12000)
    if saved['scheduler']['base_lrs'] != [policy['lr']]:
        raise ValueError('Completed scheduler base LR differs from fixed initial LR')
    return dict(passed=True, actual_initialization_and_sample_stream=stream, final_rng_and_sampler_exact=True,
        actual_training_observation_draws=384000, actual_training_candidate_path_states=1536000,
        dev_selection_opportunities=48, fixed_last_train_reserved_requests=189,
        fixed_last_train_reserved_path_states=756, new_qwen_encodings=0,
        learning_rate_first=used[0], learning_rate_last_used=used[-1], learning_rate_next_unused=saved['scheduler']['_last_lr'],
        comparison_scope='Same data, initialization, sampling, loss, architecture, K4,12000 updates and48 DEV opportunities; only LR schedule differs.')


def evaluate_fixed_last_train(model_folder, data_folder, output):
    """Same fixed-last diagnostic with an explicit cosine checkpoint identity."""
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
    _,gate=verify_export(data_folder)
    checkpoint_sha,export_sha=digest(model_folder/'last.pt'),digest(data_folder/'export_manifest.json')
    recovered=shared.recover_sealed_diagnostic(output,checkpoint_sha,export_sha)
    if recovered is not None:return recovered
    saved=torch.load(model_folder/'last.pt',map_location='cpu',weights_only=False);config=saved['config']
    if saved['step']!=12000 or config.get('cosine_protocol')!=PROTOCOL:raise ValueError('Fixed cosine last12000 required')
    model=ObservedGeometryRouteHead(**head_options(config)).eval();model.load_state_dict(saved['model'],strict=True)
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
    from scripts import train_observed_two_row as ordinary
    project,source=Path(args.project).resolve(),Path(__file__).resolve().parents[1]
    policy_path=Path(args.policy).resolve();policy=validate_policy(json.loads(policy_path.read_text()))
    output=project/policy['default_output'];model=output/'peak_seed0'
    data,selection,quality,reference,gate,sources=preflight(project,source,policy)
    metadata=dict(cosine_protocol=PROTOCOL,cosine_driver_sha256=digest(__file__),cosine_policy_sha256=digest(policy_path),
        cosine_source_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        cosine_reference_sha256=policy['reference_last_sha256'],cosine_schedule=SCHEDULE,cosine_total_steps=12000)
    frozen=dict(protocol=PROTOCOL,policy=policy,metadata=metadata,source_sha256=sources)
    checkpoint=None
    if args.resume:
        if not (output/'cosine_manifest.json').is_file():raise ValueError('No original cosine registration')
        checkpoint=torch.load(model/'last.pt',map_location='cpu',weights_only=False)
        validate_resume(dict(metadata,steps=12000),checkpoint,12000)
    elif model.exists() or (output/'cosine_manifest.json').exists():
        raise FileExistsError('Fresh independent cosine output required; use explicit same-protocol resume')
    output.mkdir(parents=True,exist_ok=True)
    shared.write_once_equal(output/'cosine_manifest.json',frozen)
    write(output/'preflight_gate.json',gate)
    event=dict(resume=args.resume,start_step=checkpoint['step'] if checkpoint else 0,status='running',device=args.device)
    started=time.perf_counter()
    try:
        complete=checkpoint is not None and checkpoint['step']==12000
        event['optimization_skipped_for_metadata_recovery']=complete
        if not complete:
            values=SimpleNamespace(data=str(data),selection=str(selection),quality_audit=str(quality),output=str(model),
                device=args.device,resume=args.resume,stop_after=None)
            with schedule_adapter(ordinary,metadata,policy['expected_sample_stream_audit']):ordinary.train(values)
        saved=torch.load(model/'last.pt',map_location='cpu',weights_only=False)
        shared.verify_ordinary_completion(model,saved['config'],ordinary.prediction_artifact_index)
        reference_saved=torch.load(reference/'last.pt',map_location='cpu',weights_only=False)
        result=json.loads((model/'summary.json').read_text())
        paired=validate_finished(policy,saved,reference_saved,result)
        write(output/'paired_stream_receipt.json',paired)
        trace=dict(protocol=SCHEDULE,total_steps=12000,base_lrs=saved['scheduler']['base_lrs'],
            rows=[dict(step=i+1,optimizer_lr=lr) for i,lr in enumerate(saved['scheduler']['used_lrs'])],
            next_lr_unused=saved['scheduler']['_last_lr'])
        shared.write_once_equal(output/'actual_learning_rates.json',trace)
        diagnostic=evaluate_fixed_last_train(model,data,output/'fixed_last_train')
        # Recheck original/source/export hashes after the complete run, never change old artifacts.
        preflight(project,source,policy)
        write(output/'cosine_result_receipt.json',dict(paired,protocol=PROTOCOL,
            checkpoint_sha256={name:digest(model/name) for name in ('best.pt','last.pt')},
            summary_sha256=digest(model/'summary.json'),original_constant_preserved=True,
            actual_learning_rates_sha256=digest(output/'actual_learning_rates.json'),
            prediction_artifacts=ordinary.prediction_artifact_index(model),fixed_last_train_diagnostic=diagnostic,
            actual_base_loop_elapsed_s=result['elapsed_s'],actual_base_loop_gpu_hours_reserved=result['gpu_hours_reserved'],
            cost_scope='Outer process/runtime events include initialization, evaluation and CPU diagnostic; base cost is nested, not additive.',
            scope='One standard optimization control; no novel route-set mechanism or further schedule sweep.'))
        event.update(status='completed',end_step=saved['step'])
    except BaseException as exc:
        event.update(status='failed',exception=repr(exc),
            unsaved_optimizer_steps_upper_bound=250,
            unsaved_candidate_path_states_upper_bound=250*32*4,
            failure_budget_scope='Conservative one-checkpoint-interval upper bound if failure followed updates; no automatic replay. Runtime is still charged.')
        raise
    finally:
        event['elapsed_seconds']=time.perf_counter()-started
        event['gpu_hours_reserved']=event['elapsed_seconds']/3600 if args.device.startswith('cuda') else 0.
        with (output/'runtime_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')


def train(args):
    policy=validate_policy(json.loads(Path(args.policy).read_text()))
    output=Path(args.project).resolve()/policy['default_output'];output.mkdir(parents=True,exist_ok=True)
    lock=output/'cosine_pipeline.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:_train(args)
    finally:lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='/home/wzy/dpvlm/route_set_v1')
    p.add_argument('--policy',default=str(Path(__file__).resolve().parents[1]/'configs/observed_two_row_cosine_v1.json'))
    p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda')
    train(p.parse_args())


if __name__=='__main__':main()
