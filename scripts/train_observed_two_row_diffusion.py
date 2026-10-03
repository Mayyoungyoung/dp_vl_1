"""Fresh observed diffusion baseline. Train/repeats/last-TRAIN never chain.

No historical model or training loop is changed. Prediction arrays and noise
are sealed before the original two-row evaluator can open verification labels.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from routeset.observed_qwen_continuation import (RequestJournal, atomic_torch_save,
    canonical, check_config, composite_evaluator, digest, exclusive_lock,
    recovered_elapsed, should_pause, write_json)

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = 'observed_two_row_diffusion_driver_v1'
SOURCE_FILES = ('scripts/train_observed_two_row_diffusion.py',
    'routeset/observed_route_diffusion.py', 'routeset/observed_diffusion_stream.py',
    'routeset/observed_qwen_continuation.py', 'routeset/observed_training_audit.py',
    'routeset/qwen_prefix_replay.py',
    'routeset/train_v2.py', 'routeset/models.py', 'routeset/common.py',
    'routeset/observed_geometry.py', 'routeset/observed_route_head.py',
    'scripts/train_observed_two_row.py', 'scripts/train_observed_geometry.py',
    'scripts/train_observed_routes.py', 'scripts/evaluate_observed_two_row.py',
    'scripts/evaluate_observed_obstacles.py', 'scripts/run_observed_two_row_composite.py',
    'scripts/export_two_row_composite_observations.py',
    'configs/observed_two_row_diffusion_v1.json')
POOL_FILES = {'predictions.npz', 'initial_noise.npz', 'generation.json',
              'metrics.json', 'per_scene.json'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_policy(policy):
    fixed=dict(protocol='observed_two_row_diffusion_v1',arms=['independent','set'],seed=0,
        feature_dim=4096,horizon=24,candidates=4,width=128,depth=2,point_width=64,
        pixel_stride=2,pooling='both',anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',
        endpoint_residual_bound=.05,event_scale=.2,parameterization='x0',diffusion_steps=100,
        sampling_steps=40,ddim_eta=0.,clip_x0=None,steps=12000,batch_size=32,lr=.0003,
        weight_decay=.0001,gradient_clip=1.,grounding_weight=.02,grounding_sigma=.025,
        grounding_target='endpoint',eval_every=250,checkpoint_every=25,eval_batch_size=1,
        eval_noise_seeds=[300000,300001,300002],training_input_draws=384000,
        training_path_states=1536000,dev_selections=48,train_inputs=285,train_references=1663,
        train_parents=95,requested_train_parents=96,dev_inputs=36,dev_parents=12,dev_references=205,
        extension_dev_raw_allowed=False)
    if policy!=read(PROJECT/'configs/observed_two_row_diffusion_v1.json') or any(policy.get(k)!=v for k,v in fixed.items()):
        raise ValueError('Only the complete fixed observed diffusion policy is supported')
    return policy


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(canonical(dict(shape=list(value.shape), dtype=str(value.dtype)))
                          + value.tobytes()).hexdigest()


def model_digest(model):
    from routeset.observed_training_audit import tensor_state_digest
    return tensor_state_digest(model.state_dict())


def evaluation_noise(ids, seed, candidates=4):
    """Evaluation has its own RNG, unchanged by training, arm, or checkpoint."""
    ids = list(map(str, ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Nonempty unique evaluation identities required')
    noise = np.random.default_rng(seed).standard_normal((len(ids), candidates, 23, 4)).astype(np.float32)
    identity = dict(ids=ids, seed=int(seed), sha256=array_digest(noise))
    return noise, identity


def evaluation_keys(identity):
    prefix = '{}:{}:r{}'.format(identity['kind'], identity['step'], identity['repeat'])
    for identifier in identity['ids']:
        key = prefix + ':' + identifier
        yield 'eval_geometry', key
        for i in range(40):
            yield 'eval_denoise', key + ':' + str(i)


def verify_pool(folder, identity):
    folder = Path(folder)
    receipt = read(folder/'pool_receipt.json')
    if receipt['identity'] != identity:
        raise ValueError('Sealed pool identity mismatch')
    names = set(receipt['artifacts'])
    if not POOL_FILES.issubset(names) or names-POOL_FILES-{'paired_language_predictions.npz'}:
        raise ValueError('Incomplete or unexpected sealed pool artifacts')
    for name, value in receipt['artifacts'].items():
        if Path(name).name != name or digest(folder/name) != value:
            raise ValueError('Sealed pool artifact changed')
    if read(folder/'metrics.json') != receipt['metrics']:
        raise ValueError('Metrics differ from original sealed receipt')
    return receipt


def reconcile_pool(journal, saved, receipt):
    if receipt['journal_before'] != saved or receipt['journal_after'] != journal.snapshot():
        raise ValueError('No exact sealed evaluation journal span')
    actual = [(r['kind'], r['key']) for r in journal.records[saved['records']:]]
    if actual != list(evaluation_keys(receipt['identity'])):
        raise ValueError('Uncheckpointed calls are not the complete known evaluation')


def accept_selection(trainer, receipt, pool, receipt_sha):
    step = receipt['identity']['step']
    if any(row['step'] == step for row in trainer.history):
        raise ValueError('Repeated model-selection opportunity')
    score = receipt['metrics']['selection_score']
    if not math.isfinite(score):
        raise ValueError('Finite original selection score required')
    trainer.history.append(dict(step=step, score=score, pool=pool,
        pool_receipt_sha256=receipt_sha, metrics=receipt['metrics']))
    if trainer.best is None or score > trainer.best['score']:
        trainer.best = dict(step=step, score=score, pool=pool)
        return True
    return False


class DiffusionTrainer:
    """One tested batch update; issued work cannot roll back across a crash."""
    def __init__(self, model, schedule, stream, config):
        import torch
        self.model, self.schedule, self.stream, self.config = model, schedule, stream, config
        self.rng = np.random.default_rng(config['seed'])
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['lr'],
            weight_decay=config['weight_decay'], betas=(.9, .999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.step, self.history, self.best = 0, [], None
        self.gradient_audit = {}
        self.recovered_sealed_evaluation_seconds = 0.

    def train_step(self, loss_function, journal):
        import torch
        step = self.step + 1
        if step > self.config['steps']:
            raise ValueError('Training budget exhausted')
        draw = self.stream.draw(self.config['batch_size'])
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        loss, details = loss_function(draw, step, journal)
        if loss.ndim or not bool(torch.isfinite(loss)):
            raise ValueError('Finite scalar training loss required')
        loss.backward()
        if step <= 2 or step % self.config['checkpoint_every'] == 0:
            self.gradient_audit = {name: dict(present=p.grad is not None,
                nonzero=p.grad is not None and bool(p.grad.count_nonzero()),
                finite=p.grad is None or bool(torch.isfinite(p.grad).all()),
                norm=float(p.grad.double().norm()) if p.grad is not None else None)
                for name, p in self.model.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config['gradient_clip'], error_if_nonfinite=True)
        journal.issue('optimizer', step)
        self.optimizer.step(); self.scheduler.step(); self.step = step
        self.optimizer.zero_grad(set_to_none=True)
        return dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm),
            learning_rate=self.optimizer.param_groups[0]['lr'], **details)

    def state_dict(self, journal, elapsed):
        from routeset.qwen_prefix_replay import cpu_copy
        from routeset.train_v2 import rng_state
        return dict(protocol=PROTOCOL, config=self.config, model=cpu_copy(self.model.state_dict()),
            optimizer=cpu_copy(self.optimizer.state_dict()), scheduler=self.scheduler.state_dict(),
            rng=rng_state(self.rng), stream=self.stream.state_dict(), stream_audit=self.stream.audit(),
            schedule=self.schedule.audit(), step=self.step, history=self.history, best=self.best,
            observation_draws=self.step*self.config['batch_size'],
            gradient_path_slots=self.step*self.config['batch_size']*self.config['candidates'],
            gradient_audit=self.gradient_audit, journal=journal.snapshot(), pending_gradients=False,
            elapsed_seconds=elapsed, recovered_sealed_evaluation_seconds=self.recovered_sealed_evaluation_seconds)

    def load_state_dict(self, state, journal, allow_sealed_evaluation=False):
        from routeset.train_v2 import restore_rng
        check_config(self.config, state['config'])
        step = state['step']; draws = step*self.config['batch_size']
        if (state['protocol'] != PROTOCOL or state['pending_gradients'] is not False or
                not 0 <= step <= self.config['steps'] or state['observation_draws'] != draws or
                state['gradient_path_slots'] != draws*self.config['candidates'] or
                state['schedule'] != self.schedule.audit()):
            raise ValueError('Incomplete or mismatched diffusion checkpoint')
        if not allow_sealed_evaluation:
            journal.require_boundary(state['journal'])
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler'])
        if (any(g['lr'] != self.config['lr'] or g['weight_decay'] != self.config['weight_decay'] for g in self.optimizer.param_groups)
                or self.scheduler.base_lrs != [self.config['lr']] or self.scheduler.last_epoch != step):
            raise ValueError('Optimizer/scheduler constant policy differs')
        self.stream.load_state_dict(state['stream'])
        audit=self.stream.audit()
        if (audit != state['stream_audit'] or audit['batches']!=step or
                audit['observation_draws']!=draws or audit['target_path_states']!=draws*self.config['candidates']):
            raise ValueError('Stream state or actual exposure changed')
        restore_rng(state['rng'], self.rng)
        self.step, self.history, self.best = step, state['history'], state['best']
        self.gradient_audit = state['gradient_audit']
        self.recovered_sealed_evaluation_seconds = state['recovered_sealed_evaluation_seconds']


def pool_identity(trainer, ids, kind, repeat, seed):
    _, noise_identity = evaluation_noise(ids, seed)
    return dict(protocol=PROTOCOL, config=trainer.config, step=trainer.step, kind=kind,
        repeat=repeat, ids=list(map(str, ids)), requests=len(ids), candidates=4,
        final_path_states=len(ids)*4, denoiser_calls=len(ids)*40,
        intermediate_path_states_including_final=len(ids)*4*40,
        model_sha256=model_digest(trainer.model), schedule=trainer.schedule.audit(), noise=noise_identity)


@contextmanager
def isolated_evaluation_rng(trainer):
    from routeset.train_v2 import rng_state, restore_rng
    saved = rng_state(trainer.rng)
    try:
        yield
    finally:
        restore_rng(saved, trainer.rng)


def evaluate_pool(trainer, data, geometry, ids, data_root, folder, kind, repeat, seed, journal, device):
    """One geometry/forty denoise calls; verification follows pool sealing.

    The dataset loader already holds supervision for training; none is passed
    to encode_observation or sample. Verification/checking uses sealed outputs.
    """
    import torch
    from scripts import train_observed_geometry as base
    from scripts import train_observed_two_row as ordinary
    from scripts.export_two_row_composite_observations import verify_export
    folder = Path(folder); ids = np.asarray(ids, dtype=int)
    identity = pool_identity(trainer, data['scene_ids'][ids], kind, repeat, seed)
    if folder.exists():
        return verify_pool(folder, identity)
    staging = folder.with_name(folder.name+'.staging')
    if staging.exists():
        raise ValueError('Incomplete issued pool preserved; automatic retry prohibited')
    staging.mkdir(parents=True)
    before = journal.snapshot(); started = synchronize()
    noise, noise_identity = evaluation_noise(data['scene_ids'][ids], seed)
    np.savez_compressed(staging/'initial_noise.npz', ids=data['scene_ids'][ids], initial_noise=noise)
    paths = np.full((len(ids),4,24,3), np.nan, np.float32)
    opened = np.full((len(ids),4,24), np.nan, np.float32)
    anchors = np.full((len(ids),3), np.nan, np.float32)
    generation = []
    def seal_predictions():
        np.savez_compressed(staging/'predictions.npz', paths=paths, gripper_open=opened,
            learned_surface_anchor=anchors, scene_ids=data['scene_ids'][ids], parent_ids=data['parent_ids'][ids])
    trainer.model.eval()
    try:
        with torch.no_grad(), isolated_evaluation_rng(trainer):
            for position, index in enumerate(ids):
                key = '{}:{}:r{}:{}'.format(kind, trainer.step, repeat, data['scene_ids'][index])
                begin = synchronize()
                inputs = base.batch_inputs(data, geometry, np.asarray([index]), device)
                journal.issue('eval_geometry', key)
                encoded = trainer.model.encode_observation(inputs)
                geometry_end = synchronize(); calls = []
                def before_denoise(i, t):
                    if i != len(calls) or i >= 40:
                        raise ValueError('Sampler exceeded or reordered the fixed40 calls')
                    journal.issue('eval_denoise', key+':'+str(i)); calls.append((i,int(t)))
                xyz, event, budget = trainer.schedule.sample(trainer.model, encoded,
                    torch.as_tensor(noise[position:position+1], device=device), before_denoise=before_denoise)
                end = synchronize()
                if len(calls) != 40 or xyz.shape != (1,4,24,3) or event.shape != (1,4,24):
                    raise ValueError('Sampler violated exact K4/H24/40-call contract')
                paths[position] = xyz.cpu().numpy()[0]; opened[position] = event.cpu().numpy()[0]
                anchors[position] = encoded['anchor_xyz'].cpu().numpy()[0]
                generation.append(dict(id=str(data['scene_ids'][index]), noise_sha256=array_digest(noise[position]),
                    geometry_and_input_seconds=geometry_end-begin, denoising_seconds=end-geometry_end,
                    cached_sampler_seconds=end-begin, actual_calls=calls, sampler_receipt=budget,
                    finite_candidates=np.isfinite(paths[position]).all((1,2)).tolist()))
        seal_predictions()
        sealed_predictions_sha = digest(staging/'predictions.npz')
        write_json(staging/'generation.json', dict(identity=identity, requests=generation,
            predictions_sha256=sealed_predictions_sha, noise=noise_identity, all_predictions_sealed_before_labels=True))
    except BaseException as exc:
        seal_predictions()
        write_json(staging/'failure.json', dict(exception=repr(exc), completed_requests=len(generation),
            requested_requests=len(ids), requested_candidates=4*len(ids), journal=journal.snapshot(),
            predictions_sha256=digest(staging/'predictions.npz'), retried=False))
        raise
    # These unchanged metric functions can now read labels; no forward remains.
    check_started = synchronize()
    metrics, rows = ordinary.observation_metrics(paths, opened, data, ids)
    control, control_arrays = ordinary.reused_language_control(paths, opened, data, geometry, ids)
    metrics['paired_language_control'] = control
    with composite_evaluator(ordinary, verify_export):
        ordinary.add_two_row_metrics(metrics, rows, paths, opened, data, ids,
            (Path(data_root)/'observations.jsonl',Path(data_root)/'supervision.jsonl'))
    metrics.update(selection_score=base.checkpoint_selection_score(metrics, 'tip_unique_valid'),
        checkpoint_selection_protocol='dev_two_row_tip_unique_valid_v1',
        checkpoint_selection_criterion='UniqueClassifiedTipValidAtK + .05 * TipValidAtK',
        generation_budget=dict(final_candidates=4, evaluated_requests=len(ids),
            evaluated_complete_path_states=4*len(ids), denoiser_calls=40*len(ids),
            intermediate_path_states_including_final=160*len(ids),
            candidate_filtering=False, language_control_additional_requests=0, repeat=repeat),
        latency_scope='Cached Qwen condition, input/geometry,40 DDIM calls; not online Qwen E2E')
    if control_arrays is not None:
        np.savez_compressed(staging/'paired_language_predictions.npz', **control_arrays)
    write_json(staging/'metrics.json', metrics); write_json(staging/'per_scene.json', rows)
    if digest(staging/'predictions.npz') != sealed_predictions_sha or pool_identity(trainer,data['scene_ids'][ids],kind,repeat,seed) != identity:
        raise ValueError('Post-generation checking changed predictions or model')
    receipt = dict(identity=identity, journal_before=before, journal_after=journal.snapshot(), metrics=metrics,
        elapsed_seconds=synchronize()-started, checking_and_serialization_seconds=synchronize()-check_started,
        artifacts={p.name:digest(p) for p in staging.iterdir() if p.is_file()})
    reconcile_pool(journal, before, receipt)
    write_json(staging/'pool_receipt.json', receipt); staging.rename(folder)
    return receipt


def verify_history(root, history, config):
    for row in history:
        folder = Path(root)/row['pool']
        if folder.resolve().parent != (Path(root)/'dev').resolve() or digest(folder/'pool_receipt.json') != row['pool_receipt_sha256']:
            raise ValueError('Historical selection pool path/hash changed')
        receipt = read(folder/'pool_receipt.json')
        if receipt['identity']['config'] != config or receipt['identity']['step'] != row['step']:
            raise ValueError('Historical selection identity changed')
        verify_pool(folder, receipt['identity'])
    if history:
        scores = [r['score'] for r in history]
        if any(not math.isfinite(v) for v in scores):
            raise ValueError('Nonfinite historical score')


def require_runtime():
    import torch
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or
            os.environ.get('RESEARCH_GPU_UUID') != 'GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab' or
            not hasattr(os,'sched_getaffinity') or len(os.sched_getaffinity(0)) != 1 or
            torch.__version__.split('+')[0] != '2.4.1'):
        raise ValueError('Fixed Torch2.4.1, GPU1/35%, and one CPU affinity required')
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35,0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    return dict(torch=torch.__version__, cpu_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
                gpu_uuid=os.environ['RESEARCH_GPU_UUID'], gpu_memory_fraction=.35)


def read_inputs(args, policy):
    from scripts import run_observed_two_row_composite as composite
    from routeset.observed_route_head import load_observed_dataset
    from scripts import train_observed_geometry as base
    root = Path(args.data)
    for name, path in [('export_manifest_sha256',root/'export_manifest.json'),
        ('quality_sha256',Path(args.quality_audit)), ('cache_receipt_sha256',root/'qwen_cache/composite_cache_receipt.json')]:
        if digest(path) != policy[name]:
            raise ValueError('Fixed composite identity differs: '+name)
    original = read(Path(args.ordinary_run)/'composite_training_receipt.json')
    if (original['export_sha256'] != policy['export_manifest_sha256'] or original['quality_sha256'] != policy['quality_sha256'] or
            original['cache_receipt_sha256'] != policy['cache_receipt_sha256']):
        raise ValueError('Original ordinary control used different data')
    for name,value in original['source_sha256'].items():
        if digest(PROJECT/name) != value:
            raise ValueError('Historical input/model/evaluation source changed: '+name)
    composite.validate_quality(root,args.quality_audit); composite.validate_cache(root)
    data = load_observed_dataset(root/'observations.jsonl',root/'supervision.jsonl',root/'qwen_cache',24,'both')
    train = np.flatnonzero(data['splits']=='TRAIN'); dev = np.flatnonzero(data['splits']=='DEV_MODEL')
    if (len(train)!=285 or len(dev)!=36 or len(data['scene_ids'])!=321 or
            not data['path_mask'][train].any(1).all() or int(data['path_mask'][train].sum())!=1663):
        raise ValueError('Exact unfiltered TRAIN285/1663 references and old DEV36 required')
    geometry = base.load_geometry(data,root/'observations.jsonl',root/'supervision.jsonl',2)
    return data,geometry,train,dev,original


def training_loss(trainer, data, geometry, device):
    import torch
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.observed_route_diffusion import targets_to_state
    from scripts import train_observed_geometry as base
    def loss(draw, step, journal):
        ids, refs = draw['indices'], draw['reference_indices']
        inputs = base.batch_inputs(data,geometry,ids,device)
        target_paths = torch.as_tensor(data['paths'][ids[:,None],refs],device=device)
        target_events = torch.as_tensor(data['events'][ids[:,None],refs],device=device)
        clean = targets_to_state(target_paths,target_events,inputs['current'])
        t = draw['timesteps'].to(device); epsilon = draw['epsilon'].to(device)
        noisy = trainer.schedule.q_sample(clean,t,epsilon)
        journal.issue('train_geometry',step)
        encoded = trainer.model.encode_observation(inputs)
        journal.issue('train_denoise',step)
        prediction = trainer.model.forward_x0(noisy,t,encoded)
        route = (prediction-clean).square().mean()
        ground = positive_endpoint_attention_loss(encoded['attention'],inputs['world_xyz'],inputs['valid_mask'],
            torch.as_tensor(data['paths'][ids,:,-1],device=device),
            torch.as_tensor(data['path_mask'][ids],device=device),trainer.config['grounding_sigma'])
        return route+trainer.config['grounding_weight']*ground, dict(route_x0_mse=float(route.detach()),grounding_loss=float(ground.detach()))
    return loss


def expected_train_calls(steps=12000, dev_count=36, eval_every=250):
    return dict(train_geometry=steps,train_denoise=steps,optimizer=steps,
                eval_geometry=(steps//eval_every)*dev_count,eval_denoise=(steps//eval_every)*dev_count*40)


def run(args):
    import torch
    from routeset.common import seed_all
    from routeset.observed_route_diffusion import ObservedRouteDiffusion,ObservedX0Schedule
    from routeset.observed_diffusion_stream import PairedPositiveStream
    started=time.perf_counter(); out=Path(args.output)
    if args.stage!='train' and (args.resume or args.stop_after is not None):
        raise ValueError('Only train supports administrative resume/pause')
    if out.exists() and any(out.iterdir()) and not (args.stage=='train' and args.resume):
        raise FileExistsError('Fresh independent stage output required')
    out.mkdir(parents=True,exist_ok=True)
    status=dict(protocol=PROTOCOL,stage=args.stage,arm=args.arm,status='running',pid=os.getpid())
    with exclusive_lock(out/'active.lock'):
        write_json(out/'status.json',status)
        try:
            policy=validate_policy(read(args.config)); runtime=require_runtime()
            preparation_start=synchronize()
            data,geometry,train_ids,dev_ids,original=read_inputs(args,policy)
            preparation_seconds=synchronize()-preparation_start
            config=dict(policy,arm=args.arm,dataset_fingerprint=data['fingerprint'],geometry_fingerprint=geometry['fingerprint'],
                ordinary_receipt_sha256=digest(Path(args.ordinary_run)/'composite_training_receipt.json'),
                ordinary_actual_index_chain_sha256=original['actual_index_chain_sha256'],
                source_sha256={p:digest(PROJECT/p) for p in SOURCE_FILES},runtime=runtime,
                code_commit=os.environ.get('CODE_COMMIT','unrecorded'))
            if config['code_commit']!=PROJECT.name:
                raise ValueError('Run only from the named immutable source release')
            seed_all(config['seed'])
            model=ObservedRouteDiffusion(set_attention=args.arm=='set').to('cuda')
            schedule=ObservedX0Schedule(); stream=PairedPositiveStream(data,train_ids,seed=config['seed'],k=4)
            if stream.audit()['initial_sampler_state_sha256']!=config['ordinary_initial_sampler_state_sha256']:
                raise ValueError('Initial observation sampler differs from ordinary control')
            trainer=DiffusionTrainer(model,schedule,stream,config)
            journal=RequestJournal(out/'requests.jsonl')
            if args.stage!='train':
                diagnostic(args,out,trainer,data,geometry,train_ids,dev_ids,journal)
                status['status']='completed';return
            if (out/'summary.json').exists():
                raise ValueError('Completed training cannot resume or silently issue work')
            prior=0.
            if args.resume:
                checkpoint=torch.load(out/'last.pt',map_location='cuda',weights_only=False)
                check_config(config,read(out/'config.json'))
                pending=checkpoint['step']>0 and checkpoint['step']%config['eval_every']==0 and not any(r['step']==checkpoint['step'] for r in checkpoint['history'])
                folder=out/'dev'/('step%05d'%checkpoint['step'])
                sealed=pending and folder.exists()
                if sealed:
                    receipt=read(folder/'pool_receipt.json');verify_pool(folder,receipt['identity'])
                    reconcile_pool(journal,checkpoint['journal'],receipt)
                    prior=recovered_elapsed(checkpoint['elapsed_seconds'],receipt['elapsed_seconds'])
                else:
                    prior=checkpoint['elapsed_seconds']
                trainer.load_state_dict(checkpoint,journal,allow_sealed_evaluation=sealed)
                if sealed:
                    if receipt['identity']!=pool_identity(trainer,data['scene_ids'][dev_ids],'dev',0,config['eval_noise_seeds'][0]):
                        raise ValueError('Recovered sealed pool was generated by a different checkpoint')
                    trainer.recovered_sealed_evaluation_seconds+=receipt['elapsed_seconds']
                verify_history(out,trainer.history,config)
            else:
                write_json(out/'config.json',config)
                write_json(out/'initialization.json',dict(model_sha256=model_digest(model),audit=model.initialization_audit,
                    schedule=schedule.audit(),stream=stream.audit(),parameters=sum(p.numel() for p in model.parameters()),
                    active_parameters=model.active_parameter_count()))
            def elapsed():return prior+time.perf_counter()-started
            def save():atomic_torch_save(out/'last.pt',trainer.state_dict(journal,elapsed()))
            def select():
                folder=out/'dev'/('step%05d'%trainer.step)
                receipt=evaluate_pool(trainer,data,geometry,dev_ids,args.data,folder,'dev',0,config['eval_noise_seeds'][0],journal,'cuda')
                if accept_selection(trainer,receipt,folder.relative_to(out).as_posix(),digest(folder/'pool_receipt.json')):
                    atomic_torch_save(out/'best.pt',trainer.state_dict(journal,elapsed()))
                save();write_json(out/'history.json',trainer.history)
            if not args.resume:save()
            if trainer.step and trainer.step%config['eval_every']==0 and not any(r['step']==trainer.step for r in trainer.history):select()
            if should_pause(trainer.step,args.stop_after,config['steps']):
                save();status.update(status='paused',step=trainer.step,actual_calls=journal.snapshot());return
            loss_function=training_loss(trainer,data,geometry,'cuda')
            while trainer.step<config['steps']:
                row=trainer.train_step(loss_function,journal)
                if trainer.step<=2 or trainer.step%25==0:
                    with (out/'training_history.jsonl').open('a',encoding='utf-8') as f:
                        f.write(json.dumps(dict(row,elapsed_seconds=elapsed(),stream=stream.audit()),allow_nan=False)+'\n')
                    print(json.dumps(row),flush=True)
                if trainer.step%config['eval_every']==0:
                    save();select()
                elif trainer.step%config['checkpoint_every']==0:save()
                if should_pause(trainer.step,args.stop_after,config['steps']):
                    save();status.update(status='paused',step=trainer.step,actual_calls=journal.snapshot());return
            if ([r['step'] for r in trainer.history]!=list(range(250,12001,250)) or journal.counts!=expected_train_calls()):
                raise ValueError('Completed logical budget and48 exact DEV pools required')
            if stream.audit()['ordinary_index_chain_sha256']!=config['ordinary_actual_index_chain_sha256']:
                raise ValueError('Actual observation draws differ from ordinary control')
            save();verify_history(out,trainer.history,config)
            best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            if best['step']!=trainer.best['step']:
                raise ValueError('Best checkpoint selection differs')
            from routeset.observed_training_audit import tensor_state_digest
            selected=read(out/trainer.best['pool']/'pool_receipt.json')
            if tensor_state_digest(best['model'])!=selected['identity']['model_sha256']:
                raise ValueError('Selected weight tensors differ from sealed best pool')
            write_json(out/'summary.json',dict(protocol=PROTOCOL,status='completed',arm=args.arm,
                best=trainer.best,last_step=trainer.step,best_metrics=next(r['metrics'] for r in trainer.history if r['step']==trainer.best['step']),
                last_metrics=trainer.history[-1]['metrics'],history=trainer.history,stream=stream.audit(),
                actual_calls=journal.snapshot(),best_checkpoint_sha256=digest(out/'best.pt'),last_checkpoint_sha256=digest(out/'last.pt'),
                config_sha256=digest(out/'config.json'),initialization_sha256=digest(out/'initialization.json'),
                observation_draws=384000,gradient_path_slots=1536000,selection_opportunities=48,
                preparation_seconds=preparation_seconds,elapsed_seconds=elapsed(),gpu_hours_reserved=elapsed()/3600,
                recovered_sealed_evaluation_seconds=trainer.recovered_sealed_evaluation_seconds,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),gradient_audit=trainer.gradient_audit,
                fixed_last_train_automatically_run=False,repeats_automatically_run=False,
                cost_scope='Full training process and cached-condition sampler; historical Qwen encoding separately charged; E2E unmeasured'))
            status['status']='completed'
        except BaseException as exc:
            status.update(status='failed',exception=repr(exc));raise
        finally:
            status['process_elapsed_seconds']=time.perf_counter()-started
            write_json(out/'status.json',status)


def diagnostic(args,out,trainer,data,geometry,train_ids,dev_ids,journal):
    import torch
    run=Path(args.train_run); summary=read(run/'summary.json')
    if (summary.get('status')!='completed' or summary['last_step']!=12000 or summary['arm']!=args.arm or
            read(run/'status.json').get('status')!='completed' or digest(run/'config.json')!=summary['config_sha256']):
        raise ValueError('Independent diagnosis requires completed12000 training')
    check_config(trainer.config,read(run/'config.json'));verify_history(run,summary['history'],trainer.config)
    original_journal=RequestJournal(run/'requests.jsonl')
    if (original_journal.snapshot()!=summary['actual_calls'] or original_journal.counts!=expected_train_calls() or
            [r['step'] for r in summary['history']]!=list(range(250,12001,250))):
        raise ValueError('Exact completed training and48 selection budgets required')
    expected_best=max(summary['history'],key=lambda r:r['score'])
    if summary['best']['step']!=expected_best['step']:
        raise ValueError('Original strict earliest-best selection was changed')
    names=['last'] if args.stage in ('fixed-last-train','denoising-diagnostic') else ['best','last']
    repeat=0 if args.stage in ('fixed-last-train','denoising-diagnostic') else int(args.stage[-1])
    results={};seen={};started=synchronize()
    for name in names:
        path=run/(name+'.pt')
        if digest(path)!=summary[name+'_checkpoint_sha256']:
            raise ValueError('Completed checkpoint bytes changed')
        state=torch.load(path,map_location='cuda',weights_only=False)
        check_config(trainer.config,state['config'])
        trainer.model.load_state_dict(state['model'],strict=True);trainer.step=state['step']
        if trainer.step!=(summary['best']['step'] if name=='best' else 12000):
            raise ValueError('Checkpoint step changed')
        state_hash=model_digest(trainer.model)
        original_pool=run/('dev/step%05d'%trainer.step)
        original_identity=read(original_pool/'pool_receipt.json')['identity']
        if state_hash!=original_identity['model_sha256']:
            raise ValueError('Checkpoint tensors differ from their original selection pool')
        if args.stage=='denoising-diagnostic':
            results[name]=denoising_diagnostic(trainer,data,geometry,train_ids,out,journal,'cuda')
            continue
        identity_key=(trainer.step,state_hash)
        if identity_key in seen:
            results[name]=dict(reused_checkpoint_pool=seen[identity_key],new_requests=0)
            continue
        ids=train_ids if args.stage=='fixed-last-train' else dev_ids
        kind='train_diag' if args.stage=='fixed-last-train' else 'dev_repeat'
        receipt=evaluate_pool(trainer,data,geometry,ids,args.data,out/name,kind,repeat,
            trainer.config['eval_noise_seeds'][repeat],journal,'cuda')
        results[name]=dict(pool=name,receipt_sha256=digest(out/name/'pool_receipt.json'),receipt=receipt)
        seen[identity_key]=name
    write_json(out/'summary.json',dict(protocol=PROTOCOL,status='completed',stage=args.stage,arm=args.arm,
        source_training_summary_sha256=digest(run/'summary.json'),results=results,actual_calls=journal.snapshot(),
        elapsed_seconds=synchronize()-started,selection_changed=False,pools_merged=False,
                scope='Independent saved-checkpoint sampler evaluation; repeat0 remains original selection pool'))


def denoising_diagnostic(trainer,data,geometry,train_ids,out,journal,device):
    """TRAIN-positive teacher noise diagnostic; never a generation-quality pool."""
    import torch
    from routeset.observed_route_diffusion import targets_to_state,decode_state
    from scripts import train_observed_geometry as base
    ids=np.asarray(train_ids[:6],dtype=int)
    if len(ids)!=6 or any(data['splits'][i]!='TRAIN' for i in ids):
        raise ValueError('Exactly first6 TRAIN inputs required')
    initial=model_digest(trainer.model); rng=np.random.default_rng(400000)
    rows=[]; tensors=[];trainer.model.eval();started=synchronize()
    with torch.no_grad(),isolated_evaluation_rng(trainer):
        for index in ids:
            identifier=str(data['scene_ids'][index]);references=np.flatnonzero(data['path_mask'][index])
            if not len(references):raise ValueError('Positive reference required')
            chosen=np.resize(references,4)
            inputs=base.batch_inputs(data,geometry,np.asarray([index]),device)
            journal.issue('diagnostic_geometry',identifier)
            encoded=trainer.model.encode_observation(inputs)
            paths=torch.as_tensor(data['paths'][index,chosen][None],device=device)
            events=torch.as_tensor(data['events'][index,chosen][None],device=device)
            clean=targets_to_state(paths,events,inputs['current'])
            for t in (0,25,50,75,99):
                noise=rng.standard_normal((1,4,23,4)).astype(np.float32)
                timestep=torch.tensor([t],device=device,dtype=torch.long)
                noisy=trainer.schedule.q_sample(clean,timestep,torch.as_tensor(noise,device=device))
                journal.issue('diagnostic_denoise',identifier+':'+str(t))
                predicted=trainer.model.forward_x0(noisy,timestep,encoded)
                predicted_paths,predicted_events=decode_state(predicted,inputs['current'])
                delta=predicted-clean
                rows.append(dict(id=identifier,reference_indices=chosen.tolist(),t=t,noise_sha256=array_digest(noise),
                    x0_mse=float(delta.square().mean()),xyz_rmse_m=float(delta[...,:3].square().mean().sqrt()),
                    endpoint_error_m=float((predicted_paths[:,:,-1]-paths[:,:,-1]).norm(dim=-1).mean()),
                    event_mae=float((predicted_events[:,:,1:]-events[:,:,1:]).abs().mean())))
                tensors.append(predicted.cpu().numpy())
    if model_digest(trainer.model)!=initial or journal.counts!=dict(diagnostic_geometry=6,diagnostic_denoise=30):
        raise ValueError('Teacher diagnostic changed weights or fixed budget')
    np.savez_compressed(Path(out)/'teacher_x0_predictions.npz',predictions=np.concatenate(tensors),
        ids=np.asarray([r['id'] for r in rows]),timesteps=np.asarray([r['t'] for r in rows]))
    result=dict(scope='Fixed positive-reference noisy x0 diagnostic, not normal generation quality',
        rows=rows,first_train_ids=list(map(str,data['scene_ids'][ids])),geometry_calls=6,denoiser_calls=30,
        intermediate_path_states=120,optimizer_steps=0,dev_inputs=0,elapsed_seconds=synchronize()-started,
        predictions_sha256=digest(Path(out)/'teacher_x0_predictions.npz'),actual_calls=journal.snapshot())
    write_json(Path(out)/'teacher_diagnostic.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['train','repeat1','repeat2','fixed-last-train','denoising-diagnostic'],required=True)
    p.add_argument('--arm',choices=['independent','set'],required=True)
    for name in ('data','quality-audit','ordinary-run','config','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--train-run');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    args=p.parse_args()
    if args.stage!='train' and not args.train_run:p.error('--train-run required for independent diagnostic stage')
    run(args)


if __name__=='__main__':main()
