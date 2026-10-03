"""Role-isolated probability experiments; each CLI stage has a fresh output.

The parent is fixed ordinary last12000. Export only selected TRAIN parents;
new DEV32 and reserved TEST payloads are never opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

ROOT = Path('/home/wzy/dpvlm/route_set_v1')
SOURCE = Path(__file__).resolve().parents[1]
POLICY = SOURCE/'configs/observed_probability_v1.json'
DATA = ROOT/'data/observed_probability_v2'
RUN = ROOT/'runs/observed_probability_v1'
PARENT = ROOT/'runs/observed_two_row_composite108_v1/peak_seed0'
OLD = ROOT/'data/observation_two_row_composite108_v1'
EXT = ROOT/'data/observed_two_row_extension288_v1'
GENERATOR = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def contained(path, folder):
    # The preserved shared runtime is Python3.8 (no Path.is_relative_to).
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except ValueError:
        return False


def role_for_index(index):
    for role, (start, end) in read(POLICY)['extension_ranges_half_open'].items():
        if start <= index < end:
            return role
    raise ValueError('Parent outside prospective extension role reservations')


def reserve_and_export():
    """Only export score-training/development now; calibration opens later."""
    p = read(POLICY)
    if sha(PARENT/'last.pt') != p['parent_checkpoint_sha256']:
        raise ValueError('Fixed parent checkpoint SHA mismatch')
    from scripts import collect_two_row_extension as ext
    from scripts import collect_two_row_formal as old
    registration, _ = ext.verify_corpus(EXT)
    gate = ext.live_layout_gate(EXT)
    oldreg, _ = old.verify_corpus(ROOT/'data/observed_two_row_formal116_v1')
    # Hash-only cross-role overlap audit, before selected payloads are read.
    groups = {}
    for origin, plans in [('extension', registration['parent_plan']), ('old', oldreg['parent_plan'])]:
        for plan in plans:
            groups.setdefault(plan['registered_geometry_1mm_sha256'], []).append((origin, plan['parent_id']))
    duplicates = [g for g in groups.values() if len(g) > 1]
    if duplicates:
        raise ValueError('Cross-role registered geometry duplicates: '+str(duplicates))
    roles = {str(i): role_for_index(i) for i in range(32, 256)}
    actual_old = {r['parent_id'] for r in lines(OLD/'observations.jsonl') if r['split']=='TRAIN'}
    held = {'two_row_reach_%d' % (400000+i) for i in range(32, 160)}
    if held & actual_old or len(actual_old) != 95:
        raise ValueError('Historical generator training population overlaps held roles')
    if DATA.exists():
        raise FileExistsError(DATA)
    DATA.mkdir(parents=True)
    write(DATA/'reservation.json', dict(policy=p, policy_sha256=sha(POLICY), extension_roles=roles,
        historical_generator_parents=sorted(actual_old), gate=gate, duplicate_groups=duplicates,
        parent_checkpoint_sha256=sha(PARENT/'last.pt'), historical_config_sha256=sha(PARENT/'config.json'),
        extension_registration_sha256=sha(EXT/'registration.json'), historical_quality_seen=True,
        note='Prospective model-role reservation after collection QA, not an untouched final test.'))
    for role in ('SCORE_TRAIN', 'DEV_SCORE'):
        export_role(role)


def export_role(role):
    """Reuse hash-indexed collector observations; no trajectory filtering."""
    if role not in ('SCORE_TRAIN', 'DEV_SCORE', 'CALIBRATION', 'FUTURE_GENERATOR_TRAIN'):
        raise ValueError('Invalid role')
    reservation = read(DATA/'reservation.json')
    if reservation['policy_sha256'] != sha(POLICY) or reservation['extension_registration_sha256'] != sha(EXT/'registration.json'):
        raise ValueError('Reservation changed')
    if role == 'CALIBRATION' and not (RUN/'q_seed0/summary.json').exists():
        raise ValueError('Freeze scorer before calibration export')
    registration = read(EXT/'registration.json')
    output = DATA/role
    output.mkdir(exist_ok=False)
    inputs, labels, inventories, hashes = [], [], [], {}
    for index in range(*read(POLICY)['extension_ranges_half_open'][role]):
        plan = registration['parent_plan'][index]
        if plan['role'] != 'TRAIN' or plan['index'] != index:
            raise ValueError('Only original TRAIN payloads permitted')
        if plan['parent_id'] in reservation['gate']['blocked_parent_ids']:
            raise ValueError('Blocked geometry parent')
        parent = plan['parent_id']
        folder = EXT/'parents/TRAIN'/parent
        if folder.is_symlink() or folder.resolve() != folder:
            raise ValueError('Unexpected raw parent path')
        artifacts = read(folder/'artifact_hashes.json')
        hashes[str(folder/'artifact_hashes.json')] = sha(folder/'artifact_hashes.json')
        closure = read(EXT/'closures'/('%03d.json' % index))
        if closure['actual_geometry_1mm_sha256'] != plan['registered_geometry_1mm_sha256']:
            raise ValueError('Measured geometry identity mismatch')
        def checked(name):
            path = (folder/name).resolve()
            if not contained(path,folder) or name not in artifacts or sha(path) != artifacts[name]:
                raise ValueError('Raw artifact changed or escaped parent: '+name)
            hashes[str(path)] = artifacts[name]
            return str(path)
        rows = lines(checked('observations.jsonl'))
        source_labels = {r['id']: r for r in lines(checked('supervision.jsonl'))}
        attempts = lines(checked('attempts.jsonl'))
        inventories.append(dict(parent_id=parent, role=role, requested_inputs=3, actual_inputs=len(rows),
                                requested_attempts=27, attempts=len(attempts), closure=closure))
        for row in rows:
            if set(row) != {'id','parent_id','split','image','instruction'} or row['parent_id'] != parent or row['split'] != 'TRAIN':
                raise ValueError('Observation whitelist or identity changed')
            original = source_labels.get(row['id'])
            target = int(row['id'].rsplit('target', 1)[1])
            positive = sorted([r for r in attempts if r['input_id']==row['id'] and r['success']], key=lambda r:r['attempt'])
            routes = [checked(parent+'/'+r['trajectory']['file']) for r in positive]
            types = [r['actual_route_type'] for r in positive]
            if original and [str((folder/f).resolve()) for f in original['routes']] != routes:
                raise ValueError('Positive reference index mismatch')
            cfg = checked('route_configs/'+parent+'.json')
            labels.append(dict(id=row['id'],parent_id=parent,split=role,task='rlbench_derived_two_row_reach',
                observation=checked(parent+'/observation.npz'), routes=routes,route_types=types,
                verification_only=checked(parent+'/verification_only.npz'),route_config=cfg,route_config_sha256=sha(cfg),
                semantic_targets=dict(centers=plan['config']['goal_xyz'],target_index=target,tolerance=.03)))
            inputs.append(dict(row, split=role, image=checked(row['image'])))
    for name, rows in [('observations', inputs), ('supervision', labels)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(output/'manifest.json', dict(role=role, source_files_sha256=hashes,parents=inventories,
        reservation_sha256=sha(DATA/'reservation.json'), requested_inputs=3*len(inventories), actual_inputs=len(inputs),
        output_files_sha256={name:sha(output/name) for name in ('observations.jsonl','supervision.jsonl')},
        locked_or_new_dev_opened=False, missing_reference_is_not_negative=True))
    print(json.dumps(dict(exported=role, parents=len(inventories), inputs=len(inputs))), flush=True)


def verify_role(role):
    path = DATA/role
    m = read(path/'manifest.json')
    for file, h in m['source_files_sha256'].items():
        if sha(file) != h:
            raise ValueError('Source changed: '+file)
    for file, h in m['output_files_sha256'].items():
        if sha(path/file) != h:
            raise ValueError('Role export changed')
    if m['reservation_sha256'] != sha(DATA/'reservation.json'):
        raise ValueError('Reservation changed')
    rows = lines(path/'observations.jsonl')
    if any(r['split'] != role or role_for_index(int(r['parent_id'].rsplit('_',1)[1])-400000) != role for r in rows):
        raise ValueError('Cross-role payload')
    return path


def torch_setup():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Authorized GPU1 only')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.35)
    return torch


def model_load():
    torch = torch_setup()
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from scripts.evaluate_observed_two_row_online import head_options
    if GENERATOR is not None:
        from routeset.observed_probability import ProbabilisticGeometryRouteHead
        checkpoint=torch.load(GENERATOR/'last.pt',map_location='cpu',weights_only=False)
        summary=read(GENERATOR/'summary.json')
        if sha(GENERATOR/'last.pt')!=summary['last_checkpoint_sha256']:raise ValueError('M8 generator changed')
        opts=head_options(read(PARENT/'config.json'));opts['max_candidates']=8
        model=ProbabilisticGeometryRouteHead(**opts).cuda().eval()
        model.load_state_dict(checkpoint['model']);model.requires_grad_(False)
        return model
    if sha(PARENT/'last.pt') != read(POLICY)['parent_checkpoint_sha256']:
        raise ValueError('Parent changed')
    checkpoint = torch.load(PARENT/'last.pt', map_location='cpu', weights_only=False)
    model = ObservedGeometryRouteHead(**head_options(read(PARENT/'config.json'))).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.requires_grad_(False)
    return model


def observation_rows(role):
    if role == 'DEV_MODEL':
        path = OLD
        inputs = [r for r in lines(path/'observations.jsonl') if r['split']=='DEV_MODEL']
        expected = {'two_row_reach_%d'%i for i in range(283264,283276)}
        if {r['parent_id'] for r in inputs} != expected:
            raise ValueError('Only old12 DEV_MODEL permitted')
    else:
        path = verify_role(role)
        inputs = lines(path/'observations.jsonl')
    labels = {r['id']:r for r in lines(path/'supervision.jsonl') if r['id'] in {x['id'] for x in inputs}}
    cache = path/'qwen_cache'
    cfg = read(cache/'cache_config.json')
    if cfg['manifest_sha256'] != sha(path/'observations.jsonl') or cfg['model_trainable_parameter_count'] != 0:
        raise ValueError('Frozen cache provenance changed')
    if cfg['revision'] != '89644892e4d85e24eaac8bacfd4f463576704203':
        raise ValueError('Pinned Qwen revision required')
    return inputs, labels, cache


def pool(role):
    import torch
    from routeset.observed_probability import route_observation_features
    from scripts.train_observed_geometry import read_geometry
    from scripts.evaluate_observed_two_row import scene_metrics
    model = model_load()
    output = RUN/'pools'/role
    output.mkdir(parents=True, exist_ok=False)
    rows, labels, cache = observation_rows(role)
    values = {key:[] for key in ('nodes','context','paths','events','labels','parents','ids','pi')}
    hashes = {}
    summaries = []
    started = time.perf_counter()
    for row in rows:
        label = labels[row['id']]
        key = hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
        file = cache/(key+'.npz')
        with np.load(file,allow_pickle=False) as a:
            if any(str(a[k].item()) != row[k] for k in ('id','parent_id','split')) or str(a['image_sha256'].item()) != sha(row['image']):
                raise ValueError('Qwen cache row/image mismatch')
            feat = np.concatenate([a['mean_hidden'].reshape(-1),a['last_hidden'].reshape(-1)]).astype('float32')
        geo = read_geometry(row['image'],label['observation'],2)
        with np.load(label['observation'],allow_pickle=False) as a:
            current = np.r_[a['gripper_pose'],np.asarray(a['gripper_open']).reshape(1)].astype('float32')
        inputs = dict(features=torch.tensor(feat[None],device='cuda'),current=torch.tensor(current[None],device='cuda'),
                      **{k:torch.tensor(v[None],device='cuda') for k,v in geo.items()})
        with torch.no_grad():
            xyz, event, details = model(**inputs)
            geometry = model.geometry(**inputs,return_point_features=True)
            context = model.head.feature_encoder(inputs['features'])+model.head.state_encoder(inputs['current'])+geometry['context']
            node, ctx = route_observation_features(xyz,event,inputs['current'],inputs['world_xyz'],inputs['rgb'],
                inputs['valid_mask'],geometry['point_features'],context,geometry['anchor_xyz'])
        xyz,event,node,ctx = [x[0].cpu().numpy() for x in (xyz,event,node,ctx)]
        # Seal actual prediction before reading geometry/semantic checking labels.
        predfile = output/(row['id']+'.npz')
        np.savez_compressed(predfile,paths=xyz,events=event)
        with np.load(label['verification_only'],allow_pickle=False) as a:
            truth = {k:a[k] for k in ('obstacle_centers','obstacle_halfsizes')}
        metrics, candidates = scene_metrics(xyz,event,dict(gripper_pose=current[:7],gripper_open=current[7]),
            truth,label['semantic_targets'],label['route_types'],read(label['route_config']))
        y = np.array([c['TipValid'] for c in candidates],dtype='float32')
        pi=details['pi'][0].cpu().numpy() if 'pi' in details else np.full(len(xyz),1/len(xyz))
        for k,v in dict(nodes=node,context=ctx,paths=xyz,events=event,labels=y,parents=row['parent_id'],ids=row['id'],pi=pi).items():
            values[k].append(v)
        summaries.append(dict(id=row['id'],metrics=metrics,candidates=candidates,prediction_sha256=sha(predfile)))
        for f in (file,row['image'],label['observation'],label['verification_only'],label['route_config']):
            hashes[str(f)] = sha(f)
    np.savez_compressed(output/'pool.npz',**{k:np.array(v) for k,v in values.items()})
    write(output/'receipt.json', dict(role=role,requests=len(rows),complete_path_states=len(rows)*model.head.max_candidates,
        parent_checkpoint_sha256=sha((GENERATOR or PARENT)/'last.pt'),source_files_sha256=hashes,pool_sha256=sha(output/'pool.npz'),
        elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        repeated_geometry_passes=len(rows),new_qwen_requests=0,no_prediction_repair=True))
    write(output/'per_scene.json',summaries)
    print(json.dumps(dict(role=role,requests=len(rows),valid=float(np.array(values['labels']).mean()))),flush=True)


def load_pool(role):
    path = RUN/'pools'/role
    receipt = read(path/'receipt.json')
    if sha(path/'pool.npz') != receipt['pool_sha256']:
        raise ValueError('Saved pool changed')
    with np.load(path/'pool.npz',allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def train_q(seed):
    torch = torch_setup()
    from routeset.observed_probability import RouteValidityHead, selection_metrics
    from routeset.train_v2 import atomic_checkpoint, rng_state
    from torch.nn import functional as F
    cfg = read(POLICY)['scorer']
    output = RUN/('q_seed%d'%seed)
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    rng = np.random.default_rng(seed)
    tr, dv = load_pool('SCORE_TRAIN'),load_pool('DEV_SCORE')
    if set(tr['parents']) & set(dv['parents']):
        raise ValueError('Score roles overlap')
    # Normalization fit only on SCORE_TRAIN, saved for deployment.
    norm = {}
    for key in ('nodes','context'):
        axes = tuple(range(tr[key].ndim-1))
        norm[key+'_mean'] = tr[key].mean(axes)
        norm[key+'_std'] = np.maximum(tr[key].std(axes),.01)
    def pack(data):
        return [torch.tensor((data[k]-norm[k+'_mean'])/norm[k+'_std'],device='cuda') for k in ('nodes','context')]
    tx,dx = pack(tr),pack(dv)
    ty,dy = [torch.tensor(d['labels'],device='cuda') for d in (tr,dv)]
    model = RouteValidityHead(tr['nodes'].shape[-1],tr['context'].shape[-1],cfg['width']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    best = float('inf');history=[];started=time.perf_counter();draw_hash=hashlib.sha256()
    for step in range(1,cfg['steps']+1):
        model.train();ids=rng.integers(len(ty),size=cfg['batch_size']);draw_hash.update(ids.astype('<i8').tobytes())
        pred=model(tx[0][ids],tx[1][ids]);loss=F.binary_cross_entropy_with_logits(pred,ty[ids])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite q loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step % cfg['eval_every']==0:
            model.eval()
            with torch.no_grad():
                logits=model(*dx);val=float(F.binary_cross_entropy_with_logits(logits,dy))
            history.append(dict(step=step,train_bce=float(loss),dev_score_bce=val))
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng),normalization=norm,
                step=step,config=cfg,seed=seed,pool_hashes={r:sha(RUN/'pools'/r/'pool.npz') for r in ('SCORE_TRAIN','DEV_SCORE')},
                draw_sha256=draw_hash.hexdigest(),history=history,elapsed_seconds=time.perf_counter()-started)
            atomic_checkpoint(output/'last.pt',state)
            if val<best:
                best=val;atomic_checkpoint(output/'best.pt',state)
            print(json.dumps(history[-1]),flush=True)
    model.load_state_dict(torch.load(output/'best.pt',map_location='cuda',weights_only=False)['model']);model.eval()
    with torch.no_grad():logits=model(*dx).cpu().numpy()
    metrics=selection_metrics(logits,dv['labels'],dv['paths'],dv['parents'])
    prevalence=float(tr['labels'].mean());null_brier=float(((dv['labels']-prevalence)**2).mean())
    gate=metrics['selected_valid']>max(metrics['first_valid'],metrics['random_expected_valid']) and metrics['brier']<null_brier
    np.savez_compressed(output/'dev_score_predictions.npz',logits=logits,labels=dv['labels'],parents=dv['parents'],ids=dv['ids'])
    write(output/'summary.json',dict(seed=seed,metrics=metrics,constant_train_prevalence_brier=null_brier,
        replication_gate_passed=gate,best_checkpoint_sha256=sha(output/'best.pt'),last_checkpoint_sha256=sha(output/'last.pt'),
        elapsed_seconds=time.perf_counter()-started,updates=cfg['steps'],candidate_score_training_exposures=cfg['steps']*cfg['batch_size']*tr['labels'].shape[1],
        history=history,peak_allocated_bytes=torch.cuda.max_memory_allocated(),generated_new_paths=0))


def calibrate(seed):
    torch = torch_setup()
    from routeset.observed_probability import RouteValidityHead,selection_metrics
    from scipy.optimize import minimize_scalar
    state=torch.load(RUN/('q_seed%d'%seed)/'best.pt',map_location='cpu',weights_only=False)
    norm=state['normalization'];cal=load_pool('CALIBRATION');dev=load_pool('DEV_MODEL')
    model=RouteValidityHead(cal['nodes'].shape[-1],cal['context'].shape[-1]).cuda().eval();model.load_state_dict(state['model'])
    def infer(data):
        with torch.no_grad():
            x=[torch.tensor((data[k]-norm[k+'_mean'])/norm[k+'_std'],device='cuda') for k in ('nodes','context')]
            return model(*x).cpu().numpy()
    clog,dlog=infer(cal),infer(dev)
    def nll(log_t):
        z=clog/np.exp(log_t)
        return float((np.logaddexp(0,z)-cal['labels']*z).mean())
    fit=minimize_scalar(nll,bounds=(-3,3),method='bounded')
    temperature=float(np.exp(fit.x))
    output=RUN/('calibration_seed%d'%seed);output.mkdir(exist_ok=False)
    np.savez_compressed(output/'predictions.npz',calibration_logits=clog,dev_logits=dlog,dev_labels=dev['labels'],dev_parents=dev['parents'])
    write(output/'summary.json',dict(temperature=temperature,scorer_checkpoint_sha256=sha(RUN/('q_seed%d'%seed)/'best.pt'),
        calibration_parents=len(set(cal['parents'])),calibration_fit_success=bool(fit.success),
        calibration_nll_before=nll(0),calibration_nll_after=nll(fit.x),
        dev_model_uncalibrated=selection_metrics(dlog,dev['labels'],dev['paths'],dev['parents']),
        dev_model_calibrated=selection_metrics(dlog,dev['labels'],dev['paths'],dev['parents'],temperature),
        note='Old DEV_MODEL is reused development evidence. Temperature cannot change ranking. No final-test claim.'))


def package(seed):
    torch=torch_setup()
    from routeset.observed_probability import RouteValidityHead,ScoredRoutePlanner
    from scripts.train_observed_geometry import read_geometry
    output=RUN/('deployment_seed%d'%seed);output.mkdir(exist_ok=False)
    scorer_file=RUN/('q_seed%d'%seed)/'best.pt'
    state=torch.load(scorer_file,map_location='cpu',weights_only=False)
    calibration=read(RUN/('calibration_seed%d'%seed)/'summary.json')
    if calibration['scorer_checkpoint_sha256']!=sha(scorer_file):raise ValueError('Calibrator/scorer mismatch')
    generator=model_load()
    for role in ('SCORE_TRAIN','DEV_SCORE'):
        receipt=read(RUN/'pools'/role/'receipt.json')
        if receipt['parent_checkpoint_sha256']!=sha((GENERATOR or PARENT)/'last.pt') or receipt['pool_sha256']!=state['pool_hashes'][role]:
            raise ValueError('Scorer belongs to another generator/pool')
    scorer=RouteValidityHead().cuda().eval();scorer.load_state_dict(state['model'])
    planner=ScoredRoutePlanner(generator,scorer,state['normalization'],calibration['temperature']).cuda().eval()
    rows,labels,cache=observation_rows('DEV_MODEL');row=rows[0];label=labels[row['id']]
    key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
    with np.load(cache/(key+'.npz'),allow_pickle=False) as a:feat=np.r_[a['mean_hidden'].reshape(-1),a['last_hidden'].reshape(-1)].astype('float32')
    geo=read_geometry(row['image'],label['observation'],2)
    with np.load(label['observation'],allow_pickle=False) as a:current=np.r_[a['gripper_pose'],np.asarray(a['gripper_open']).reshape(1)].astype('float32')
    inputs=dict(features=torch.tensor(feat[None],device='cuda'),current=torch.tensor(current[None],device='cuda'),
                **{k:torch.tensor(v[None],device='cuda') for k,v in geo.items()})
    tic=time.perf_counter()
    with torch.no_grad():prediction=planner(**inputs,return_k=4)
    torch.cuda.synchronize();seconds=time.perf_counter()-tic
    pool=load_pool('DEV_MODEL');pos=int(np.flatnonzero(pool['ids']==row['id'])[0])
    with np.load(RUN/('calibration_seed%d'%seed)/'predictions.npz',allow_pickle=False) as a:logits=a['dev_logits'][pos]
    expected=1/(1+np.exp(-logits/calibration['temperature']))
    actual=prediction['q'][0].cpu().numpy();paths=prediction['paths'][0].cpu().numpy()
    if not np.allclose(paths,pool['paths'][pos],atol=1e-6,rtol=1e-5) or not np.allclose(actual,expected,atol=1e-5,rtol=1e-5):
        raise ValueError('Integrated deployment differs from sealed candidate/scorer output')
    from routeset.observed_probability import select_route_indices
    selections={str(k):select_route_indices(prediction['paths'],prediction['q'],k)[0].cpu().tolist() for k in (1,2,4)}
    pi=prediction['pi'][0].cpu().numpy() if prediction['pi'] is not None else None
    np.savez_compressed(output/'example.npz',paths=paths,q=actual,pi=pi if pi is not None else np.array([]))
    torch.save(dict(planner_state=planner.state_dict(),feature_dim=4096,horizon=24,M=generator.head.max_candidates,
        anchor_mode='straight_through_peak',temperature=calibration['temperature'],has_pi=GENERATOR is not None,
        frozen_qwen_revision='89644892e4d85e24eaac8bacfd4f463576704203'),output/'planner.pt')
    write(output/'manifest.json',dict(generator_sha256=sha((GENERATOR or PARENT)/'last.pt'),scorer_sha256=sha(scorer_file),
        calibration_sha256=sha(RUN/('calibration_seed%d'%seed)/'summary.json'),planner_sha256=sha(output/'planner.pt'),
        example_id=row['id'],example_image=row['image'],example_image_sha256=sha(row['image']),q=actual.tolist(),pi=None if pi is None else pi.tolist(),
        K_selected_indices=selections,integrated_equal_to_saved_outputs=True,internal_M=generator.head.max_candidates,
        new_forward_requests=1,new_complete_path_states=generator.head.max_candidates,new_qwen_encodings=0,
        cached_feature_head_seconds=seconds,scope='One integrated cached-Qwen/RGB-D smoke request, not E2E VLM latency or robot execution',
        probability_scope='q calibrated to tip-only checker, empirical calibration does not guarantee accuracy; pi separately balanced reference mass'))


def main():
    global RUN,GENERATOR
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','export-role','pool','train-q','calibrate','package'])
    p.add_argument('--role');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--generator-family',choices=['M8_ordinary_seed0_v2','M8_balanced_probability_seed0_v2',
        'M8_ordinary_seed0_expanded','M8_balanced_probability_seed0_expanded'])
    a=p.parse_args()
    if a.generator_family:
        GENERATOR=RUN/a.generator_family
        RUN=RUN/(a.generator_family+'_scores')
    if a.stage=='export':reserve_and_export()
    elif a.stage=='export-role':export_role(a.role)
    elif a.stage=='pool':pool(a.role)
    elif a.stage=='train-q':train_q(a.seed)
    elif a.stage=='calibrate':calibrate(a.seed)
    elif a.stage=='package':package(a.seed)


if __name__=='__main__':main()
