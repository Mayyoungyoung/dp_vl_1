"""Controlled fine-tuning and ordinary-mode baselines on verified support."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from scripts.run_observed_probability import ROOT, SOURCE, read, write, sha, lines, torch_setup
from scripts.paired_modes_data import DATA, RUN as OLD_RUN
from scripts.research_v3_audit import mode, exact_clear, plain, coverage, average
from scripts.evaluate_paired_modes import references, check_candidates, FROZEN_Q
from scripts.analyze_paired_selection import select
from routeset.research_v3_modes import passage_signature

RUN=ROOT/'runs/research_v3_v1'
POLICY=SOURCE/'configs/research_v3_frequency_v1.json'
SUPPORT=RUN/'frequency_support_v1'


def prepare():
    cfg=read(POLICY); SUPPORT.mkdir(parents=True,exist_ok=False)
    obs={r['id']:r for r in lines(DATA/'export/observations.jsonl')}
    records=[]; xs=[]; es=[]; masks=[]; types=[]; hashes={}
    maxrefs=9*(1+cfg['variants_per_witness']); rng=np.random.default_rng(cfg['perturbation_seed'])
    failures=Counter(); attempted=0; accepted=0
    for label in lines(DATA/'export/supervision.jsonl'):
        assert label['split'] in ('TRAIN','DEV_MODEL')
        ref=references(label); paths=[]; events=[]; modes=[]; witnesses=[]
        # Reach events are constant; preserve the actual observed state.
        for j,(p,v) in enumerate(zip(ref['paths'],ref['reference_valid'])):
            if not v:continue
            tag=mode(p,ref['config']); assert tag is not None
            paths.append(p);events.append(np.full(24,float(np.asarray(ref['current']['gripper_open']))));modes.append(tag)
            witnesses.append(j)
            count=0; tries=0
            while count<cfg['variants_per_witness'] and tries<100:
                tries+=1;attempted+=1
                t=np.linspace(0,1,24)
                basis=np.stack([np.sin(np.pi*t),np.sin(2*np.pi*t),np.sin(3*np.pi*t)],1)
                jitter=basis.dot(rng.uniform(-1,1,(3,3)))/3*np.array(cfg['jitter_amplitudes_m'])
                candidate=p+jitter;candidate[[0,-1]]=p[[0,-1]]
                clear=exact_clear(candidate[None],ref['truth']['obstacle_centers'],ref['truth']['obstacle_halfsizes'])[0]
                floor=candidate[:,2].min()>=ref['config']['post_base_z']+.02
                same=mode(candidate,ref['config'])==tag
                if not clear or not floor or not same:
                    failures['collision']+=int(not clear);failures['floor']+=int(not floor);failures['mode_changed']+=int(not same)
                    continue
                paths.append(candidate);events.append(events[-1].copy());modes.append(tag);witnesses.append(j)
                count+=1;accepted+=1
            if count!=cfg['variants_per_witness']:raise RuntimeError('Uncompleted witness perturbation '+label['id'])
        assert len(paths)>0
        values=np.asarray(paths,dtype=np.float32);ee=np.asarray(events,dtype=np.float32)
        # Recheck actual stored precision; no float64-only validity labels.
        _,cc=check_candidates(values,ee,label,ref['current'],ref['truth'],ref['config'])
        assert all(c['TipValid'] for c in cc)
        digests=[hashlib.sha256(p.tobytes()).hexdigest() for p in values];assert len(set(digests))==len(digests)
        pad=maxrefs-len(values);xs.append(np.pad(values,((0,pad),(0,0),(0,0))))
        es.append(np.pad(ee,((0,pad),(0,0))));masks.append([True]*len(values)+[False]*pad)
        types.append(modes+['']*pad)
        records.append(dict(id=label['id'],split=label['split'],parent=label['parent_id'],
            family=label['parent_id'].rsplit('_',1)[0],reference_witness_indices=witnesses,
            modes=modes,route_sha256=digests,rgb_sha256=sha(obs[label['id']]['image']),
            observation_sha256=sha(label['observation']),route_config_sha256=sha(label['route_config'])))
        for key in ('observation','route_config','verification_only'):hashes[label[key]]=sha(label[key])
        for file in label['routes']:hashes[file]=sha(file)
    np.savez_compressed(SUPPORT/'support.npz',paths=np.array(xs),events=np.array(es),mask=np.array(masks),
        modes=np.array(types),ids=np.array([r['id'] for r in records]),splits=np.array([r['split'] for r in records]))
    write(SUPPORT/'records.json',records)
    write(SUPPORT/'receipt.json',dict(policy=cfg,policy_sha256=sha(POLICY),support_sha256=sha(SUPPORT/'support.npz'),
        attempts=attempted,accepted=accepted,rejections=failures,input_sha256=hashes,
        independent_train_families=len({r['family'] for r in records if r['split']=='TRAIN'}),
        scope='Unique geometry-verified perturbations are correlated teacher variants, not independent demonstrations'))


def probabilities(tags, arm):
    groups=sorted(set(tags));major='gap0|gap0'
    if arm in ('empirical_uniform','balanced','set_matching'):return {g:1/len(groups) for g in groups}
    p=.9 if arm=='empirical_90' else .98
    if major not in groups or len(groups)==1:raise ValueError('Registered majority witness missing')
    return {g:p if g==major else (1-p)/(len(groups)-1) for g in groups}


def pad_targets(target, targete):
    maximum=max(len(a) for a in target)
    valid=np.array([[True]*len(a)+[False]*(maximum-len(a)) for a in target])
    paths=np.array([np.pad(a,((0,maximum-len(a)),(0,0),(0,0))) for a in target])
    events=np.array([np.pad(a,((0,maximum-len(a)),(0,0))) for a in targete])
    return paths,events,valid


def record_history(step, planned_steps):
    """A pause must not change the experiment's deterministic log schedule."""
    return step % 100 == 0 or step == planned_steps


def sampled_distinct_targets(tags,rng,budget=8):
    """Ordinary mode-stratified sampling; exactly the same target slot budget."""
    groups=sorted(set(tags));order=rng.permutation(len(groups))[:budget]
    picked=[int(rng.choice(np.flatnonzero(tags==groups[j]))) for j in order]
    if len(picked)<budget:
        mass=probabilities(tags,'balanced')
        weights=np.array([mass[g]/np.count_nonzero(tags==g) for g in tags])
        picked.extend(rng.choice(len(tags),budget-len(picked),replace=True,p=weights).tolist())
    return np.array(picked,dtype=int)


def match_loss(pred,target,tags,rng):
    import torch
    from scipy.optimize import linear_sum_assignment
    cost=(pred[:, :,None]-target[:,None]).square().mean((-1,-2));terms=[]
    for i in range(len(pred)):
        groups=sorted(set(tags[i]));groups=[groups[j] for j in rng.permutation(len(groups))[:pred.shape[1]]]
        groupcost=torch.stack([cost[i,:,np.flatnonzero(tags[i]==g)].amin(-1) for g in groups],1)
        fit=cost[i].amin(1)
        # Exact injective coverage, spare hypotheses may choose any positive.
        rows,cols=linear_sum_assignment((groupcost-fit[:,None]).detach().cpu().numpy().T)
        total=fit.sum()+sum(groupcost[c,r]-fit[c] for r,c in zip(rows,cols))
        terms.append(total/pred.shape[1])
    return torch.stack(terms).mean()


def train(arm,name,steps=None,stop_after=None,resume=False,safety_control=None):
    torch=torch_setup()
    from scripts.train_paired_modes import load_data, source_identity
    from scripts.train_observed_geometry import batch_inputs
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import segment_clearance_loss
    from routeset.paired_modes import workspace_floor_loss
    from routeset.train_v2 import positive_assignment_loss,atomic_checkpoint,rng_state,restore_rng
    from routeset.observed_training_audit import tensor_state_digest,new_stream_audit,append_indices
    cfg=read(POLICY);steps=steps or cfg['steps'];out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed run cannot be overwritten')
    assert sha(SUPPORT/'support.npz')==read(SUPPORT/'receipt.json')['support_sha256']
    with np.load(SUPPORT/'support.npz') as a: support={k:a[k] for k in a.files}
    data,geo,configs,cs,hs,_,labelmap=load_data();index={str(x):i for i,x in enumerate(data['scene_ids'])}
    trainids=np.flatnonzero(support['splits']=='TRAIN');mapped=np.array([index[str(support['ids'][i])] for i in trainids])
    assert len(trainids)==1152 and all(data['splits'][j]=='TRAIN' for j in mapped)
    init=OLD_RUN/'R1_seed0/last.pt'
    safety=None
    if safety_control is not None:
        assert arm=='set_matching' and safety_control in ('mean','worst')
        safety=read(RUN/'safety_train_diagnostic_v1/RESULTS.json')
        assert safety['ordinary_safety_control_gate']
        init=RUN/'frequency_set_matching/last.pt'
        assert sha(init)==safety['checkpoint_sha256']
    saved=torch.load(init,map_location='cpu',weights_only=False)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    rng=np.random.default_rng(cfg['seed']);lrng=np.random.default_rng(cfg['seed']+6100)
    model=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda();model.load_state_dict(saved['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    settings=dict(arm=arm,steps=steps,policy=cfg,initial_checkpoint_sha256=sha(init),support_sha256=sha(SUPPORT/'support.npz'),
        data_fingerprint=geo['fingerprint'],source_sha256=source_identity(),head_options=saved['config']['head_options'],
        input_ids_sha256=hashlib.sha256(mapped.tobytes()).hexdigest())
    if safety is not None:
        settings.update(safety_control=safety_control,safety_diagnostic_sha256=sha(RUN/'safety_train_diagnostic_v1/RESULTS.json'),
            collision_coefficient=160. if safety_control=='mean' else safety['worst_collision_coefficient'],
            continuation_scope='Fresh optimizer from fixed full-set checkpoint; same1200-step input/target stream in both arms')
    audit=new_stream_audit(model,rng.bit_generator.state,torch.get_rng_state());initial=tensor_state_digest(model.state_dict())
    exposure=Counter();unique=set();history=[];start=0
    if resume:
        s=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert s['settings']==settings
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        restore_rng(s['rng'],rng);lrng.bit_generator.state=s['loss_rng'];audit=s['sampler'];exposure=Counter(s['mode_exposure'])
        unique=set(s['unique_routes']);history=s['history'];start=s['step']
    else:write(out/'config.json',settings)
    cs=torch.tensor(cs,dtype=torch.float32,device='cuda');hs=torch.tensor(hs,dtype=torch.float32,device='cuda')
    floors=torch.tensor([c['post_base_z']+.02 for c in configs],dtype=torch.float32,device='cuda')
    tic=time.monotonic();end=min(steps,stop_after or steps)
    def state(step):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        rng=rng_state(rng),loss_rng=lrng.bit_generator.state,sampler=audit,settings=settings,step=step,history=history,
        mode_exposure=dict(exposure),unique_routes=sorted(unique),config=dict(head_options=saved['config']['head_options']))
    for step in range(start+1,end+1):
        local=rng.integers(len(trainids),size=cfg['batch_size']);si=trainids[local];ids=mapped[local];audit=append_indices(audit,ids)
        inp=batch_inputs(data,geo,ids,'cuda');model.train();xyz,event,details=model(**inp)
        target=[];targete=[];sampletags=[]
        for sidx in si:
            n=int(support['mask'][sidx].sum());tags=support['modes'][sidx,:n]
            if arm=='set_matching':picked=np.arange(n)
            elif arm=='set_sampled':picked=sampled_distinct_targets(tags,lrng)
            else:
                probs=probabilities(tags,arm);w=np.array([probs[g]/np.count_nonzero(tags==g) for g in tags])
                picked=lrng.choice(n,8,replace=True,p=w)
            target.append(support['paths'][sidx,picked]);targete.append(support['events'][sidx,picked]);sampletags.append(tags[picked])
            for j in picked:
                exposure[tags[j]]+=1;unique.add('%d:%d'%(sidx,j))
        # Matching accepts variable reference counts. Pad only its batch labels.
        target,targete,valid=pad_targets(target,targete)
        tx=torch.tensor(target,device='cuda');te=torch.tensor(targete,device='cuda')
        pred=torch.cat([xyz[:,:,1:],event[:,:,1:,None]*.2],-1)
        truth=torch.cat([tx[:,:,1:],te[:,:,1:,None]*.2],-1)
        if arm=='set_matching':
            # Variable-cardinality groups are processed without padded classes.
            regression=torch.stack([match_loss(pred[i:i+1],truth[i:i+1,:len(sampletags[i])],
                np.array([sampletags[i]]),lrng) for i in range(len(pred))]).mean()
        else:regression=positive_assignment_loss(pred,truth,valid,'positive',lrng)
        ground=positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],tx[:,:,-1],
            torch.tensor(valid,device='cuda'),.025)
        if safety_control=='worst':
            from routeset.segment_clearance import path_segment_clearances
            deficits=(.02-path_segment_clearances(xyz,cs[ids],hs[ids])).clamp_min(0)
            clear=(safety['worst_collision_coefficient']/160*deficits.square().amax(-1).mean()
                +workspace_floor_loss(xyz,floors[ids]))
        else:clear=segment_clearance_loss(xyz,cs[ids],hs[ids])+workspace_floor_loss(xyz,floors[ids])
        loss=regression+.02*ground+160*clear
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pilot loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step()
        if record_history(step,steps):
            history.append(dict(step=step,loss=float(loss),regression=float(regression),clearance=float(clear)))
            print(json.dumps(history[-1]),flush=True)
        if step%300==0 or step==end:atomic_checkpoint(out/'recovery.pt',state(step))
    if end==steps:
        atomic_checkpoint(out/'last.pt',state(end));assert tensor_state_digest(model.state_dict())!=initial
        write(out/'summary.json',dict(steps=end,elapsed_seconds=time.monotonic()-tic,sampler=audit,
            last_sha256=sha(out/'last.pt'),initial_sha256=initial,final_sha256=tensor_state_digest(model.state_dict()),
            mode_exposure=dict(exposure),unique_routes=len(unique),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            exposure_scope='set matching uses full positive set per draw; actual target processing cost explicitly differs'))


def fixed_path_scores(planner, paths, events, inputs):
    """Score external paths with the untouched deployment feature extractor."""
    from routeset.observed_probability import route_observation_features
    geometry=planner.generator.geometry(**inputs,return_point_features=True)
    context=(planner.generator.head.feature_encoder(inputs['features'])
        +planner.generator.head.state_encoder(inputs['current'])+geometry['context'])
    nodes,ctx=route_observation_features(paths,events,inputs['current'],inputs['world_xyz'],
        inputs['rgb'],inputs['valid_mask'],geometry['point_features'],context,geometry['anchor_xyz'])
    logits=planner.scorer((nodes-planner.nodes_mean)/planner.nodes_std,
        (ctx-planner.context_mean)/planner.context_std)
    return (logits/planner.temperature).sigmoid()


def evaluate(name,fixed_q=False,anchor_mass=False):
    torch=torch_setup()
    from scripts.evaluate_paired_modes import inputs_for
    from routeset.observed_probability import load_scored_planner
    from scripts.paired_modes_reliability import reliability_metrics
    if anchor_mass and not fixed_q:raise ValueError('anchor intervention requires complete fixed scorer')
    folder=RUN/name;out=folder/('evaluation_anchor_mass_v1' if anchor_mass else 'evaluation_fixed_q_v2' if fixed_q else 'evaluation');out.mkdir(exist_ok=False)
    bundle=OLD_RUN/'reliability/R1_seed0/deployment_seed0/planner.pt'
    model=load_scored_planner(bundle,'cuda');s=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    if fixed_q:
        generator=copy.deepcopy(model.generator)
        generator.load_state_dict(s['model']);generator.eval();generator.requires_grad_(False)
        if anchor_mass:generator.geometry.anchor_mode='local_mass_peak'
    else:model.generator.load_state_dict(s['model'])
    model.eval();model.requires_grad_(False)
    with np.load(SUPPORT/'support.npz') as a: support={k:a[k] for k in a.files}
    smap={str(k):i for i,k in enumerate(support['ids'])}
    rows=[r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL']
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    paths=[];events=[];qs=[];details=[];hashes={};tic=time.monotonic()
    for row in rows:
        inp=inputs_for(row,labels[row['id']],DATA/'export/qwen_cache',torch,hashes)
        with torch.inference_mode():
            if fixed_q:
                xyz,event,_=generator(**inp)
                p=dict(paths=xyz,events=event,q=fixed_path_scores(model,xyz,event,inp))
            else:p=model(**inp,return_k=4)
        paths.append(p['paths'][0].cpu().numpy());events.append(p['events'][0].cpu().numpy());qs.append(p['q'][0].cpu().numpy())
    # Candidates are generated solely from observations before oracle checking.
    ys=[]
    for i,row in enumerate(rows):
        ref=references(labels[row['id']]);_,cand=check_candidates(paths[i],events[i],ref['label'],ref['current'],ref['truth'],ref['config'])
        valid=np.array([c['TipValid'] for c in cand]);ys.append(valid)
        words=[mode(p,ref['config']) if v else None for p,v in zip(paths[i],valid)]
        j=smap[row['id']];known=set(support['modes'][j,support['mask'][j]])
        rare=known-{'gap0|gap0'};selected=select(paths[i],qs[i],4)
        metrics={str(k):coverage(words,valid,known,list(range(k))) for k in (1,2,4,8)}
        rawset={w for w in words if w is not None};selset={words[j] for j in selected if valid[j] and words[j] is not None}
        details.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],variant=row['parent_id'].rsplit('_',1)[1],
            raw=metrics['8'],selected=coverage(words,valid,known,selected),budgets_prefix=metrics,
            rare_recall8=len(rawset&rare)/len(rare),rare_recall4=len(selset&rare)/len(rare),
            majority_hit=int('gap0|gap0' in rawset),invalid_fraction=float((~valid).mean()),
            duplicates=int(valid.sum())-len(rawset),unknown_valid=sum(bool(v) and w is None for v,w in zip(valid,words)),
            words=words,valid=valid.tolist(),q=qs[i].tolist(),candidates=cand))
    data=dict(paths=np.array(paths),events=np.array(events),q=np.array(qs),labels=np.array(ys),
        ids=np.array([r['id'] for r in rows]),parents=np.array([r['parent_id'] for r in rows]))
    np.savez_compressed(out/'pool.npz',**data)
    q=np.clip(data['q'],1e-7,1-1e-7); rel=reliability_metrics(np.log(q/(1-q)),data)
    # Calibrated fixed-q transfer scores; do not fit a temperature on this DEV.
    summary=dict(raw=average([r['raw'] for r in details]),selected=average([r['selected'] for r in details]),
        rare_recall8=float(np.mean([r['rare_recall8'] for r in details])),rare_recall4=float(np.mean([r['rare_recall4'] for r in details])),
        majority_hit=float(np.mean([r['majority_hit'] for r in details])),duplicates=float(np.mean([r['duplicates'] for r in details])),
        reliability=rel,by_variant={v:average([r['raw'] for r in details if r['variant']==v]) for v in ('open','closed','shifted')},
        generator_sha256=sha(folder/'last.pt'),scorer_bundle_sha256=sha(bundle),pool_sha256=sha(out/'pool.npz'),
        elapsed_seconds=time.monotonic()-tic,input_sha256=hashes,locked_access=False,
        scoring_contract='fixed complete deployment scorer and observation encoder' if fixed_q else 'fixed scorer weights with changed generator observation encoder',
        anchor_intervention='local_mass_peak sigma .025 over all valid points, inference only' if anchor_mass else None)
    write(out/'rows.json',plain(details));write(out/'metrics.json',plain(summary));print(json.dumps(summary['raw']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','evaluate_fixed_q','evaluate_anchor_mass']);p.add_argument('--arm',default='empirical_uniform')
    p.add_argument('--name');p.add_argument('--steps',type=int);p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
    p.add_argument('--safety-control',choices=['mean','worst']);a=p.parse_args()
    if a.arm not in ('empirical_uniform','empirical_90','empirical_98','balanced','set_matching','set_sampled'):
        p.error('Unregistered training arm')
    if a.stage=='prepare':prepare()
    elif a.stage=='train':train(a.arm,a.name,a.steps,a.stop_after,a.resume,a.safety_control)
    else:evaluate(a.name,fixed_q=a.stage in ('evaluate_fixed_q','evaluate_anchor_mass'),anchor_mass=a.stage=='evaluate_anchor_mass')
