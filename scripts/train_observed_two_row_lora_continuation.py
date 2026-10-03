"""Independent frozen/LoRA common-head continuation; stages never chain."""
import argparse
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from routeset.observed_qwen_continuation import (POLICY, PROTOCOL, HEAD_SHA, EXPORT_SHA,
    REVISION, RequestJournal, SerialTrainer, atomic_torch_save, check_config,
    composite_evaluator, digest, draw_plan, exclusive_lock, reconcile_sealed_pool,
    recovered_elapsed, seal_draw_plan, should_pause, validate_policy, verify_pool, write_json)

PROJECT = Path(__file__).resolve().parents[1]
PROBE_SOURCE = 'eddfacaddab2d12c67f5a56fd775de173c05f9b2'
SOURCE_FILES = ('routeset/observed_qwen_continuation.py',
    'scripts/train_observed_two_row_lora_continuation.py', 'routeset/qwen_prefix_corpus.py',
    'routeset/qwen_prefix_replay.py', 'scripts/train_observed_lora.py',
    'scripts/run_observed_two_row_composite.py', 'scripts/train_observed_two_row.py',
    'scripts/train_observed_geometry.py', 'scripts/train_observed_routes.py',
    'scripts/export_two_row_composite_observations.py', 'scripts/evaluate_observed_two_row.py',
    'scripts/audit_two_row_qwen_prefix_replay.py',
    'scripts/evaluate_observed_obstacles.py', 'routeset/observed_geometry.py',
    'routeset/observed_route_head.py', 'routeset/models.py', 'routeset/train_v2.py', 'routeset/common.py',
    'configs/observed_two_row_lora_continuation_v1.json',
    'scripts/evaluate_observed_two_row_online.py')


def synchronized():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def require_runtime():
    import torch
    import transformers
    from scripts.audit_two_row_qwen_prefix_replay import OFFICIAL_SHA
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available()
            or os.environ.get('RESEARCH_GPU_UUID') != 'GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
            or torch.__version__.split('+')[0] != '2.4.1' or transformers.__version__ != '4.57.1'
            or not hasattr(os, 'sched_getaffinity') or len(os.sched_getaffinity(0)) != 1):
        raise ValueError('Pinned runtime, GPU1/35%, and one CPU affinity required')
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35, 0)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    installed = Path(inspect.getfile(transformers)).parent
    for relative, value in OFFICIAL_SHA.items():
        if digest(installed/relative) != value:
            raise ValueError('Official Qwen runtime source changed')
    return dict(torch=torch.__version__, transformers=transformers.__version__,
                gpu_uuid=os.environ['RESEARCH_GPU_UUID'],cpu_threads=1,
                official_source_sha256=OFFICIAL_SHA)


def read_identity(args, policy):
    """All corpus/export gates before historical loader reads selected labels."""
    import torch
    from routeset.qwen_prefix_corpus import PrefixCorpus
    from scripts import run_observed_two_row_composite as composite
    if digest(args.head) != HEAD_SHA or digest(Path(args.data)/'export_manifest.json') != EXPORT_SHA:
        raise ValueError('Fixed common head/export changed')
    composite.validate_quality(args.data, args.quality_audit)
    cache = composite.validate_cache(args.data)
    head_source = torch.load(args.head, map_location='cpu', weights_only=False)
    source_receipt = json.loads((Path(args.head).parent/'composite_training_receipt.json').read_text())
    if source_receipt['checkpoint_sha256']['last.pt'] != HEAD_SHA:
        raise ValueError('Original complete-run receipt differs')
    for name, value in source_receipt['source_sha256'].items():
        if digest(PROJECT/name) != value:
            raise ValueError('Original model/evaluator/loader implementation changed: '+name)
    if head_source['step'] != 12000 or head_source['trajectory_exposures'] != 1536000:
        raise ValueError('Common source must be the completed last12000')
    cfg = head_source['config']
    for name, value in composite.MODEL_OPTIONS.items():
        if cfg.get(name) != value:
            raise ValueError('Original ordinary model setting differs: '+name)
    corpus = PrefixCorpus(args.prefix_corpus, expected_fingerprint=policy['prefix_manifest_sha256'])
    m = corpus.manifest
    if (m['export_sha256'] != EXPORT_SHA or m['model_revision'] != REVISION or
            m['processor_revision'] != REVISION or m['counts'] != dict(TRAIN=285, DEV_MODEL=36, total=321)
            or m['probe']['source_commit'] != PROBE_SOURCE or
            m['historical_cache_receipt_sha256'] != cfg['composite_cache_sha256'] or
            m['replay_source_sha256'] != digest(PROJECT/'routeset/qwen_prefix_replay.py')):
        raise ValueError('Passing prefix corpus identity differs from this experiment')
    rows = [json.loads(x) for x in (Path(args.data)/'observations.jsonl').read_text().splitlines()]
    if (tuple(r['id'] for r in rows) != tuple(corpus.ids) or
            m['observations_sha256'] != digest(Path(args.data)/'observations.jsonl')):
        raise ValueError('Exact composite321 observation order required')
    for row in rows:
        c = corpus.rows_by_id[row['id']]
        if (c['parent_id'] != row['parent_id'] or c['split'] != row['split'] or
                c['source']['observation_row'] != row or
                c['comparisons'] != dict(historical_exact=True, replay_exact=True)):
            raise ValueError('Prefix row role/input/equivalence differs')
        corpus.get(row['id'])  # Once-per-process SHA validation and CPU RAM memoization.
    model_root = Path(args.model).resolve()
    if digest(model_root/'provenance.json') != m['model_provenance_sha256']:
        raise ValueError('Model provenance changed')
    for name, value in m['model_assets_sha256'].items():
        path = (model_root/name).resolve()
        try:
            path.relative_to(model_root)
        except ValueError:
            raise ValueError('Model asset outside pinned root')
        if digest(path) != value:
            raise ValueError('Pinned Qwen asset changed')
    return corpus, rows, head_source, cache


def load_data(args, rows):
    from routeset.observed_route_head import load_observed_dataset
    from scripts import train_observed_geometry as base
    root = Path(args.data)
    data = load_observed_dataset(root/'observations.jsonl', root/'supervision.jsonl', root/'qwen_cache', 24, 'both')
    if list(map(str, data['scene_ids'])) != [r['id'] for r in rows]:
        raise ValueError('Historical loader changed/omitted input IDs')
    train = np.flatnonzero(data['splits']=='TRAIN'); dev = np.flatnonzero(data['splits']=='DEV_MODEL')
    if len(train) != 285 or len(dev) != 36 or not data['path_mask'][train].any(axis=1).all():
        raise ValueError('Fixed positive TRAIN285 and DEV36 required; no filtering')
    geometry = base.load_geometry(data, root/'observations.jsonl', root/'supervision.jsonl', 2)
    return data, geometry, train, dev


def create_models(args, checkpoint):
    import torch
    from transformers import Qwen3VLForConditionalGeneration
    from routeset.common import seed_all
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.qwen_prefix_replay import adapter_parameters, parameter_hashes
    from scripts.train_observed_lora import install_lora
    from scripts.evaluate_observed_two_row_online import head_options
    seed_all(POLICY['seed'])
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    text = backbone.model.language_model
    if len(text.layers) != 28 or text.config.hidden_size != 2048 or text.config.attention_dropout != 0:
        raise ValueError('Unexpected pinned Qwen architecture/dropout')
    seed_all(POLICY['adapter_seed'])
    if args.arm == 'lora':
        install_lora(backbone, 8, 16.)
    adapters = adapter_parameters(backbone)
    if sum(p.numel() for p in adapters.values()) != (114688 if args.arm=='lora' else 0):
        raise ValueError('Unexpected adapter count')
    if {n for n,p in backbone.named_parameters() if p.requires_grad} != set(adapters):
        raise ValueError('Unapproved trainable backbone parameter')
    seed_all(POLICY['seed'])
    head = ObservedGeometryRouteHead(**head_options(checkpoint['config'])).to('cuda')
    head.load_state_dict(checkpoint['model'], strict=True)
    if not all(p.requires_grad for p in head.parameters()):
        raise ValueError('Every existing head/geometry parameter must continue training')
    # Wrapper names change base q/v paths; compare this representation at end.
    frozen_hashes = parameter_hashes(backbone, exclude_adapters=True)
    return backbone, head, adapters, frozen_hashes


def module_hash(module):
    from routeset.qwen_prefix_replay import parameter_hashes
    import hashlib
    from routeset.observed_qwen_continuation import canonical
    return hashlib.sha256(canonical(parameter_hashes(module))).hexdigest()


def pool_identity(state, ids, step, kind):
    from routeset.qwen_prefix_replay import parameter_hashes, tensor_hash
    import hashlib
    from routeset.observed_qwen_continuation import canonical
    return dict(protocol=PROTOCOL, config=state.config, step=step, kind=kind,
        ids=list(map(str, ids)), requests=len(ids), candidate_path_states=4*len(ids),
        head_sha256=module_hash(state.head),
        adapter_sha256=hashlib.sha256(canonical({k:tensor_hash(v) for k,v in state.adapters.items()})).hexdigest())


def evaluate_pool(backbone, trainer, corpus, data, geometry, ids, args, journal, folder, kind):
    """All dynamic tails, then exactly one original-head call per ID, then checks."""
    import torch
    from routeset.qwen_prefix_replay import replay_feature
    from scripts import train_observed_two_row as ordinary
    from scripts.export_two_row_composite_observations import verify_export
    identity = pool_identity(trainer, data['scene_ids'][ids], trainer.step, kind)
    folder = Path(folder)
    if folder.exists():
        return verify_pool(folder, identity)
    staging = folder.with_name(folder.name+'.staging')
    if staging.exists():
        raise ValueError('Incomplete issued evaluation preserved; no automatic replay')
    staging.mkdir(parents=True)
    before = journal.snapshot(); started = synchronized()
    live = dict(data, features=np.full_like(data['features'], np.nan))
    prefix = 'dev' if kind=='dev' else 'train_diag'
    with torch.no_grad():
        for index in ids:
            identifier = str(data['scene_ids'][index]); key = str(trainer.step)+':'+identifier
            journal.issue(prefix+'_tail', key)
            live['features'][index] = replay_feature(backbone, corpus.get(identifier)).cpu().numpy()[0]
    if not np.isfinite(live['features'][ids]).all():
        raise ValueError('Nonfinite current-adapter condition')
    np.savez_compressed(staging/'tail_features.npz', ids=data['scene_ids'][ids], features=live['features'][ids])
    position = [0]
    def before_head(module, arguments, kwargs):
        i = position[0]
        if i >= len(ids) or kwargs['features'].shape != (1,4096):
            raise ValueError('Evaluation exceeded one B1 head per input')
        identifier = str(data['scene_ids'][ids[i]])
        if not np.array_equal(kwargs['features'].detach().cpu().numpy()[0], live['features'][ids[i]]):
            raise ValueError('Stale/evaluation feature identity mismatch')
        journal.issue(prefix+'_head', str(trainer.step)+':'+identifier); position[0] += 1
    handle = trainer.head.register_forward_pre_hook(before_head, with_kwargs=True)
    try:
        with composite_evaluator(ordinary, verify_export):
            metrics = ordinary.evaluate(trainer.head, live, geometry, ids, 'cuda', staging,
                batch_size=1, selection_metric='tip_unique_valid',
                evaluation_sources=(Path(args.data)/'observations.jsonl',Path(args.data)/'supervision.jsonl'))
    finally:
        handle.remove()
    if position[0] != len(ids) or pool_identity(trainer,data['scene_ids'][ids],trainer.step,kind) != identity:
        raise ValueError('Evaluation count or weight state changed')
    receipt = dict(identity=identity, journal_before=before, journal_after=journal.snapshot(),
        metrics=metrics, elapsed_seconds=synchronized()-started,
        artifacts={p.name:digest(p) for p in staging.iterdir() if p.is_file()},
        scope='Current-adapter prefix replay plus original evaluator; not full online Qwen latency')
    write_json(staging/'pool_receipt.json', receipt); staging.rename(folder)
    return receipt


def accept_dev(trainer, receipt, relative_pool, receipt_sha256):
    step = receipt['identity']['step']
    if any(r['step']==step for r in trainer.history):
        raise ValueError('Duplicate DEV selection')
    metrics = receipt['metrics']; score = metrics['selection_score']
    row = dict(step=step, score=score, pool=relative_pool, metrics=metrics,
               pool_receipt_sha256=receipt_sha256)
    trainer.history.append(row)
    if trainer.best is None or score > trainer.best['score']:
        trainer.best = dict(step=step, score=score, pool=relative_pool)
        return True
    return False


def source_identity(args, policy, runtime, corpus, data, geometry, plan_sha):
    return dict(policy, arm=args.arm, head_source_sha256=HEAD_SHA,
        prefix_corpus_sha256=corpus.fingerprint, dataset_fingerprint=data['fingerprint'],
        geometry_fingerprint=geometry['fingerprint'], quality_sha256=digest(args.quality_audit),
        draw_plan_sha256=plan_sha, runtime=runtime,
        source_sha256={name:digest(PROJECT/name) for name in SOURCE_FILES},
        prefix_probe=corpus.manifest['probe'], code_commit=os.environ.get('CODE_COMMIT','unrecorded'),
        cache_scope='frozen_layer26_input_only; tails and trainable geometry recomputed',
        shared_pretraining_path_states=1536000, reset_optimizer=True)


def verify_history(root, history, config):
    for row in history:
        folder=Path(root)/row['pool']
        if folder.resolve().parent != (Path(root)/'dev').resolve():
            raise ValueError('Evaluation pool outside this run')
        if digest(folder/'pool_receipt.json')!=row['pool_receipt_sha256']:
            raise ValueError('Previously selected pool receipt changed')
        receipt=json.loads((folder/'pool_receipt.json').read_text())
        if receipt['identity']['config']!=config or receipt['identity']['step']!=row['step']:
            raise ValueError('Previously selected pool config/step differs')
        verify_pool(folder,receipt['identity'])


def completed_read_only(args, out):
    """Completed output reuse does not load Qwen or issue another request."""
    summary_path=out/'summary.json'
    if not summary_path.exists():return False
    result=json.loads(summary_path.read_text())
    if result.get('status')!='completed' or result.get('arm')!=args.arm or result.get('protocol')!=PROTOCOL:
        raise ValueError('Existing output is not a matching completed run')
    if digest(args.head)!=HEAD_SHA or digest(Path(args.data)/'export_manifest.json')!=EXPORT_SHA:
        raise ValueError('Completed source identity changed')
    journal=RequestJournal(out/'requests.jsonl')
    if args.stage=='train':
        config=json.loads((out/'config.json').read_text())
        for name,value in config['source_sha256'].items():
            if digest(PROJECT/name)!=value:raise ValueError('Completed source changed')
        if digest(out/'config.json')!=result['config_sha256'] or digest(out/'last.pt')!=result['last_checkpoint_sha256'] or digest(out/'best.pt')!=result['best_checkpoint_sha256']:
            raise ValueError('Completed checkpoint/config changed')
        if result['actual_calls']!=journal.snapshot():raise ValueError('Completed journal changed')
        verify_history(out,result['history'],config)
    else:
        receipt=verify_pool(out/'pool',result['receipt']['identity'])
        if receipt!=result['receipt'] or receipt['journal_after']!=journal.snapshot():
            raise ValueError('Completed diagnostic pool/journal changed')
        config=receipt['identity']['config']
        for name,value in config['source_sha256'].items():
            if digest(PROJECT/name)!=value:raise ValueError('Completed diagnostic source changed')
    if (config['arm']!=args.arm or digest(Path(args.prefix_corpus)/'manifest.json')!=config['prefix_corpus_sha256']
            or digest(args.draw_plan)!=config['draw_plan_sha256'] or digest(args.quality_audit)!=config['quality_sha256']):
        raise ValueError('Completed cache/plan/quality identity changed')
    print(json.dumps(dict(status='completed_reused',new_forward_calls=0,output=str(out))),flush=True)
    return True


def run(args):
    import torch
    from routeset.common import seed_all
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.qwen_prefix_replay import (replay_feature, parameter_hashes, tensor_hash)
    from routeset.train_v2 import positive_assignment_loss
    from scripts import train_observed_geometry as base
    started = time.perf_counter(); out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    status = dict(protocol=PROTOCOL, stage=args.stage, arm=args.arm, status='running', pid=os.getpid())
    if args.stage=='train' and not args.resume and any(out.iterdir()):
        raise FileExistsError('Fresh arm output required')
    with exclusive_lock(out/'active.lock'):
        validate_policy(json.loads(Path(args.config).read_text()))
        if completed_read_only(args,out):return
        write_json(out/'status.json', status)
        try:
            policy = validate_policy(json.loads(Path(args.config).read_text()))
            runtime = require_runtime()
            corpus, rows, common_head, cache = read_identity(args, policy)
            data, geometry, train_ids, dev_ids = load_data(args, rows)
            plan = draw_plan(data['scene_ids'][train_ids], seed=policy['sampler_seed'])
            plan_sha = seal_draw_plan(args.draw_plan, plan)
            config = source_identity(args, policy, runtime, corpus, data, geometry, plan_sha)
            backbone, head, adapters, frozen_hashes = create_models(args, common_head)
            initial_head = module_hash(head); initial_adapters = {n:tensor_hash(p) for n,p in adapters.items()}
            initial_head_parameters = {n:tensor_hash(p) for n,p in head.named_parameters()}
            seed_all(policy['seed'])
            trainer = SerialTrainer(head, adapters, config, plan)
            journal = RequestJournal(out/'requests.jsonl')
            prior_elapsed = 0.
            elapsed = lambda: prior_elapsed + time.perf_counter()-started
            def checkpoint():
                atomic_torch_save(out/'last.pt', trainer.state_dict(journal, elapsed()))
                write_json(out/'history.json', trainer.history)
            def pause_if_requested():
                if not should_pause(trainer.step,args.stop_after,3000):return False
                checkpoint()
                status.update(status='paused',step=trainer.step,logical_draws=trainer.step*32,
                    candidate_path_states=trainer.step*128,journal=journal.snapshot(),
                    resume_note='Keep the same source/arguments; remove --stop-after and add --resume. Total budget remains3000.',
                    administrative_stop_after=args.stop_after)
                return True
            if args.stage=='fixed-last-train':
                if not args.train_run:
                    raise ValueError('Diagnostic requires completed --train-run')
                run_root = Path(args.train_run)
                final = json.loads((run_root/'summary.json').read_text())
                saved = torch.load(run_root/'last.pt', map_location='cpu', weights_only=False)
                if final['status']!='completed' or digest(run_root/'last.pt')!=final['last_checkpoint_sha256'] or saved['step']!=3000:
                    raise ValueError('Completed sealed last3000 required')
                training_journal = RequestJournal(run_root/'requests.jsonl')
                trainer.load_state_dict(saved, training_journal)
                verify_history(run_root,trainer.history,trainer.config)
                receipt = evaluate_pool(backbone,trainer,corpus,data,geometry,train_ids,args,journal,out/'pool','train')
                if journal.counts!=dict(train_diag_tail=285,train_diag_head=285) or journal.snapshot()!=receipt['journal_after']:
                    raise ValueError('Fixed-last diagnostic issued budget differs')
                if parameter_hashes(backbone,exclude_adapters=True)!=frozen_hashes:
                    raise ValueError('Frozen backbone changed during diagnostic')
                diagnostic_elapsed=time.perf_counter()-started
                write_json(out/'summary.json',dict(status='completed',protocol=PROTOCOL,arm=args.arm,
                    source_last_sha256=final['last_checkpoint_sha256'],receipt=receipt,
                    actual_requests=285,requested_requests=288,missing_requests=3,
                    tail_calls=285,head_calls=285,candidate_path_states=1140,new_qwen_full_calls=0,
                    optimizer_updates=0,elapsed_seconds=diagnostic_elapsed,
                    gpu_hours_reserved=diagnostic_elapsed/3600))
                status['status']='completed'
                return
            if args.resume:
                saved = torch.load(out/'last.pt',map_location='cpu',weights_only=False)
                check_config(config,json.loads((out/'config.json').read_text()))
                mismatch = journal.snapshot()!=saved['journal']
                trainer.load_state_dict(saved,journal,allow_sealed_evaluation=mismatch)
                should_pause(trainer.step,args.stop_after,3000)
                prior_elapsed = saved['elapsed_seconds']
                verify_history(out,trainer.history,config)
                if mismatch:
                    if trainer.step%250 or any(r['step']==trainer.step for r in trainer.history):
                        raise ValueError('Uncheckpointed training calls; automatic replay prohibited')
                    folder=out/'dev'/('step%04d'%trainer.step)
                    receipt=verify_pool(folder,pool_identity(trainer,data['scene_ids'][dev_ids],trainer.step,'dev'))
                    reconcile_sealed_pool(journal,saved['journal'],receipt)
                    prior_elapsed=recovered_elapsed(prior_elapsed,receipt['elapsed_seconds'])
                    trainer.recovered_sealed_evaluation_seconds+=receipt['elapsed_seconds']
                    improved=accept_dev(trainer,receipt,str(folder.relative_to(out)),digest(folder/'pool_receipt.json'))
                    if improved:atomic_torch_save(out/'best.pt',trainer.state_dict(journal,elapsed()))
                    checkpoint()
            else:
                if journal.records:
                    raise ValueError('Existing issued requests require explicit resume')
                write_json(out/'config.json',config)
                write_json(out/'initialization.json',dict(common_checkpoint_sha256=HEAD_SHA,
                    actual_head_sha256=initial_head,adapter_initial_sha256=initial_adapters,
                    initial_head_parameter_sha256=initial_head_parameters,
                    draw_fingerprint=plan['fingerprint'],head_parameters=sum(p.numel() for p in head.parameters()),
                    adapter_parameters=sum(p.numel() for p in adapters.values()),fresh_adam_states=True,
                    frozen_base_sha256=frozen_hashes,prefix_bytes=sum(r['bytes'] for r in corpus.manifest['rows'])))
                checkpoint()
            def select_current():
                folder=out/'dev'/('step%04d'%trainer.step)
                receipt=evaluate_pool(backbone,trainer,corpus,data,geometry,dev_ids,args,journal,folder,'dev')
                improved=accept_dev(trainer,receipt,str(folder.relative_to(out)),digest(folder/'pool_receipt.json'))
                # Best must exist before last points to it. An interruption after
                # the first atomic write is recovered only from the sealed pool.
                if improved:atomic_torch_save(out/'best.pt',trainer.state_dict(journal,elapsed()))
                checkpoint()
            if trainer.step and trainer.step%250==0 and not any(r['step']==trainer.step for r in trainer.history):
                select_current()  # Crash before issuing evaluation: do not skip a selection.
            if pause_if_requested():return
            id_lookup={str(identifier):i for i,identifier in enumerate(data['scene_ids'])}
            def loss_function(identifier,step,micro,rng):
                key='%d:%d:%s'%(step,micro,identifier);index=id_lookup[identifier]
                journal.issue('train_tail',key)
                with torch.set_grad_enabled(args.arm=='lora'):
                    feature=replay_feature(backbone,corpus.get(identifier))
                inputs=base.batch_inputs(data,geometry,np.asarray([index]),'cuda')
                inputs['features']=feature
                journal.issue('train_head',key)
                xyz,opened,details=head(**inputs)
                target_xyz=torch.as_tensor(data['paths'][index:index+1],device='cuda')
                events=torch.as_tensor(data['events'][index:index+1],device='cuda')
                prediction=torch.cat((xyz[:,:,1:],opened[:,:,1:,None]*.2),-1)
                target=torch.cat((target_xyz[:,:,1:],events[:,:,1:,None]*.2),-1)
                route=positive_assignment_loss(prediction,target,data['path_mask'][index:index+1],'saturation',rng)
                ground=positive_endpoint_attention_loss(details['attention'],inputs['world_xyz'],inputs['valid_mask'],
                    target_xyz[:,:,-1],torch.as_tensor(data['path_mask'][index:index+1],device='cuda'),.025)
                return route+.02*ground
            while trainer.step<3000:
                row=trainer.train_step(loss_function,journal)
                if trainer.step<=2 or trainer.step%25==0:
                    if trainer.step==2:
                        write_json(out/'step0002_gradient_audit.json',trainer.gradient_audit)
                        if any(not trainer.gradient_audit[n]['nonzero'] for n in adapters):
                            raise ValueError('All8 adapters need finite nonzero step2 gradients')
                    with (out/'training_history.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(row,elapsed_seconds=elapsed(),logical_draws=trainer.step*32))+'\n')
                    checkpoint()
                    print(json.dumps(dict(row,elapsed_seconds=elapsed(),logical_draws=trainer.step*32)),flush=True)
                if trainer.step%250==0:
                    select_current()
                if pause_if_requested():return
            if len(trainer.history)!=12 or [r['step'] for r in trainer.history]!=list(range(250,3001,250)):
                raise ValueError('All12 fixed DEV selection opportunities required')
            expected=dict(train_tail=96000,train_head=96000,optimizer=3000,dev_tail=432,dev_head=432)
            if journal.counts!=expected:
                raise ValueError('Actual issued budget differs; no success summary')
            verify_history(out,trainer.history,config)
            final_frozen=parameter_hashes(backbone,exclude_adapters=True)
            if final_frozen!=frozen_hashes:
                raise ValueError('Frozen backbone changed')
            final_adapters={n:tensor_hash(p) for n,p in adapters.items()}
            if adapters and not all(final_adapters[n]!=initial_adapters[n] for n in adapters):
                raise ValueError('Not every adapter tensor actually changed')
            final_head_parameters={n:tensor_hash(p) for n,p in head.named_parameters()}
            head_changed={n:final_head_parameters[n]!=initial_head_parameters[n] for n in initial_head_parameters}
            if not all(head_changed.values()):
                raise ValueError('Not all shared head parameters actually updated')
            if any(not trainer.gradient_audit[n]['nonzero'] for n in adapters):
                raise ValueError('All8 adapters need finite nonzero final-step gradients')
            checkpoint()
            best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
            if best['step']!=trainer.best['step']:
                raise ValueError('Best weight/pool selection differs')
            import hashlib
            from routeset.observed_qwen_continuation import canonical
            best_identity=json.loads((out/trainer.best['pool']/'pool_receipt.json').read_text())['identity']
            if (hashlib.sha256(canonical({n:tensor_hash(v) for n,v in best['model'].items()})).hexdigest()!=best_identity['head_sha256'] or
                    hashlib.sha256(canonical({n:tensor_hash(v) for n,v in best['adapters'].items()})).hexdigest()!=best_identity['adapter_sha256']):
                raise ValueError('Best checkpoint tensors differ from sealed selected pool')
            summary=dict(protocol=PROTOCOL,status='completed',arm=args.arm,
                common_head_sha256=HEAD_SHA,best_step=trainer.best['step'],last_step=3000,
                best_pool=trainer.best['pool'],last_pool='dev/step3000',
                best_metrics=next(r['metrics'] for r in trainer.history if r['step']==trainer.best['step']),
                last_metrics=trainer.history[-1]['metrics'],history=trainer.history,
                best_checkpoint_sha256=digest(out/'best.pt'),last_checkpoint_sha256=digest(out/'last.pt'),
                actual_calls=journal.snapshot(),observation_draws=96000,training_path_states=384000,
                shared_pretraining_path_states=1536000,per_arm_logical_cumulative_path_states=1920000,
                optimizer_steps=3000,dev_selection_opportunities=12,fixed_last_train_automatically_run=False,
                initial_head_sha256=initial_head,final_head_sha256=module_hash(head),
                initial_adapters=initial_adapters,final_adapters=final_adapters,frozen_base_unchanged=True,
                frozen_base_parameter_tensors=len(frozen_hashes),head_parameter_changed=head_changed,
                last_gradient_audit=trainer.gradient_audit,elapsed_seconds=elapsed(),
                recovered_sealed_evaluation_seconds=trainer.recovered_sealed_evaluation_seconds,
                gpu_hours_reserved=elapsed()/3600,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                new_qwen_full_calls=0,train_tail_replays=96000,dev_tail_replays=432,
                cost_scope='Shared prefix preparation separately charged; prefix evaluation is not online E2E',
                completion_proof='All logical updates returned and all12 evaluation pools sealed; failed issued calls would remain uncertified and charged',
                config_sha256=digest(out/'config.json'))
            write_json(out/'summary.json',summary);status['status']='completed'
        except BaseException as exc:
            status.update(status='failed',exception=repr(exc))
            raise
        finally:
            status['process_elapsed_seconds']=time.perf_counter()-started
            write_json(out/'status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['train','fixed-last-train'],required=True)
    p.add_argument('--arm',choices=['frozen','lora'],required=True)
    for name in ('model','data','head','prefix-corpus','draw-plan','quality-audit','config','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--train-run');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,help='Administrative absolute logical-step pause; total remains3000')
    args=p.parse_args()
    if args.stage!='train' and args.resume:
        p.error('Diagnostic uses sealed-pool reuse, not training resume')
    if args.stage!='train' and args.stop_after is not None:
        p.error('stop-after is only an administrative training boundary')
    if args.stop_after is not None and not 1<=args.stop_after<=3000:
        p.error('stop-after must lie within1..3000')
    run(args)


if __name__=='__main__':main()
