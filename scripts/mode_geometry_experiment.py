"""Frozen encoder, same-information B/C/D training and independent DEV checks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import random
import time
import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,lines,torch_setup
from scripts import research_v3_frequency as old
from scripts.paired_modes_data import DATA
from scripts.research_v3_audit import mode,plain,coverage,average
from scripts.evaluate_paired_modes import references,check_candidates
from scripts.research_v3_counterfactual import EDGES
from scripts.analyze_paired_selection import select

RUN=ROOT/'runs/mode_geometry_v1'
PREP=RUN/'prepared_v2'
INITIAL=old.RUN/'safety_mean/last.pt'
POOL=old.RUN/'verified_edit_all_modes_support_v1/support.npz'
Q=old.RUN/'matched_q_paired_v1/reliability/mean_seed0/calibration_seed0/scorer_bundle.pt'


def new_model(torch, conditional, independent_decoder=False):
    from routeset.mode_geometry import ModeGeometryHead
    saved=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model=ModeGeometryHead(**saved['config']['head_options'],conditional=conditional,decoder_communication=not independent_decoder)
    result=model.load_state_dict(saved['model'],strict=False)
    assert not result.unexpected_keys and all(k.startswith(('mode_predictor.','mode_embedding.','variant_embedding.')) for k in result.missing_keys)
    model.freeze_encoders()
    return model.cuda(),saved['config']['head_options']


def prepare():
    torch=torch_setup()
    from routeset.mode_geometry import VOCAB
    from routeset.observed_route_head import load_observed_dataset
    from scripts.train_observed_geometry import load_geometry,batch_inputs
    from scripts.train_verified_set import checker
    PREP.mkdir(parents=True,exist_ok=False)
    assert sha(POOL)==read(POOL.parent/'receipt.json')['support_sha256']
    with np.load(POOL) as z: s={k:z[k] for k in z.files}
    assert len(s['ids'])==1152 and set(s['splits'])=={'TRAIN'}
    d=load_observed_dataset(DATA/'export/observations.jsonl',DATA/'export/supervision.jsonl',DATA/'export/qwen_cache',24,'both')
    geo=load_geometry(d,DATA/'export/observations.jsonl',DATA/'export/supervision.jsonl',2)
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    meta={r['id']:r for r in lines(DATA/'export/metadata.jsonl') if r['role']=='TRAIN'}
    di={str(k):i for i,k in enumerate(d['scene_ids'])}; mapped=np.array([di[str(k)] for k in s['ids']])
    assert set(d['splits'][mapped])=={'TRAIN'}
    torch.manual_seed(0);model,_=new_model(torch,False);model.eval()
    context=[];anchor=[];parent=[]
    with torch.no_grad():
        for offset in range(0,len(mapped),32):
            inp=batch_inputs(d,geo,mapped[offset:offset+32],'cuda')
            c,a=model.encode(**inp);p,_,_=model.decode(c,a,inp['current'])
            context.extend(c.cpu().numpy());anchor.extend(a.cpu().numpy());parent.extend(p.cpu().numpy())
    n=len(mapped); evidence=np.full((n,len(VOCAB)),-1,np.float32)
    embedding_sum=np.zeros((len(VOCAB),model.head.queries.shape[-1]),np.float64);embedding_count=np.zeros(len(VOCAB))
    query=model.head.queries.detach().cpu().numpy(); refs=[];groups=defaultdict(dict); checks=[]
    bymode=[]; hashes={}
    for i,ident in enumerate(s['ids']):
        ident=str(ident);ref=references(labels[ident]);refs.append(ref);check=checker(ref);checks.append(check)
        valid=s['mask'][i]; tags=s['modes'][i,valid]; paths=s['paths'][i,valid];events=s['events'][i,valid]
        ok,words=check(paths,events);assert ok.all() and words==tags.tolist()
        bm={w:np.flatnonzero(valid & (s['modes'][i]==w)) for w in sorted(set(tags))};bymode.append(bm)
        cfg=ref['config']
        for mi,w in enumerate(VOCAB):
            # A closed inflated internal interval certifies this operational word absent.
            impossible=any(t=='gap1' and cfg['post_y'][r][1]-cfg['post_y'][r][0] <= .075 for r,t in enumerate(w.split('|')))
            if impossible:evidence[i,mi]=0
            if w in bm:
                assert not impossible;evidence[i,mi]=1
                cost=((np.asarray(parent[i])[:,None]-s['paths'][i,bm[w]][None])**2).mean((-1,-2)).min(-1)
                embedding_sum[mi]+=query[cost.argmin()];embedding_count[mi]+=1
        m=meta[ident];groups[(m['family_id'],ident.rsplit('target',1)[1])][m['variant']]=i
        for key in ('observation','route_config','verification_only'):hashes[labels[ident][key]]=sha(labels[ident][key])
    pairs=[];sources=[];targets=[];masks=[];changed=[];counts=Counter()
    for _,variants in sorted(groups.items()):
        for va,vb in EDGES:
            a,b=variants[va],variants[vb];common=sorted(set(bymode[a])&set(bymode[b]))
            pa=np.zeros((16,24,3),np.float32);pb=pa.copy();mask=np.zeros(16,bool);chg=mask.copy()
            for w in common:
                ai,bi=bymode[a][w],bymode[b][w]
                cross,_=checks[b](s['paths'][a,ai],s['events'][a,ai])
                # Prefer an actually invalidated old route if one exists; then nearest valid same-word target.
                options=ai[~cross] if (~cross).any() else ai
                distance=((s['paths'][a,options][:,None]-s['paths'][b,bi][None])**2).mean((-1,-2))
                ia,ib=np.unravel_index(distance.argmin(),distance.shape);mi=VOCAB.index(w)
                pa[mi]=s['paths'][a,options[ia]];pb[mi]=s['paths'][b,bi[ib]];mask[mi]=True;chg[mi]=bool((~cross).any())
                counts['C_invalid_old_same_mode_new_valid' if chg[mi] else 'B_surviving_coordinates']+=1
            pairs.append([a,b]);sources.append(pa);targets.append(pb);masks.append(mask);changed.append(chg)
    init=embedding_sum/np.maximum(embedding_count[:,None],1)
    np.savez_compressed(PREP/'train.npz',context=np.asarray(context),anchor=np.asarray(anchor),current=d['current'][mapped],
        ids=s['ids'],evidence=evidence,mode_init=init.astype(np.float32),pairs=np.array(pairs),
        pair_source=np.array(sources),pair_target=np.array(targets),pair_mask=np.array(masks),pair_changed=np.array(changed),
        centers=np.array([r['truth']['obstacle_centers'] for r in refs]),halves=np.array([r['truth']['obstacle_halfsizes'] for r in refs]),
        floors=np.array([r['config']['post_base_z']+.02 for r in refs],np.float32))
    write(PREP/'manifest.json',dict(counts=dict(counts),train_requests=n,pairs=len(pairs),mode_vocabulary=VOCAB,
        mode_train_counts=embedding_count.tolist(),initial_sha256=sha(INITIAL),support_sha256=sha(POOL),
        data_sha256=sha(PREP/'train.npz'),observed_fingerprint=geo['fingerprint'],input_hashes=hashes,
        evidence_counts={str(v):int((evidence==v).sum()) for v in (-1,0,1)},locked_access=False,
        note='Only TRAIN indices encoded/used for fitting. Shared dataset loader materializes permitted DEV; no DEV update. Existing RGB-D and matching frozen Qwen features reused.'))
    print(json.dumps(dict(counts=counts,pairs=len(pairs))),flush=True)


def train(arm,seed,steps,name,resume=False,stop_after=None,pair_weight=1.,independent_decoder=False):
    torch=torch_setup()
    from routeset.mode_geometry import mode_loss,pair_displacement_loss,VOCAB
    from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
    from routeset.observed_training_audit import tensor_state_digest
    from routeset.segment_clearance import segment_clearance_loss
    from routeset.paired_modes import workspace_floor_loss
    from scipy.optimize import linear_sum_assignment
    out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed output')
    manifest=read(PREP/'manifest.json');assert sha(PREP/'train.npz')==manifest['data_sha256']
    with np.load(PREP/'train.npz') as z:d={k:z[k] for k in z.files}
    assert sha(POOL)==manifest['support_sha256']
    with np.load(POOL) as z:s={k:z[k] for k in z.files}
    assert np.array_equal(s['ids'],d['ids'])
    per_mode=[{VOCAB.index(w):np.flatnonzero(s['mask'][i] & (s['modes'][i]==w))
        for w in set(s['modes'][i,s['mask'][i]])} for i in range(len(s['ids']))]
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    model,options=new_model(torch,arm!='B',independent_decoder)
    with torch.no_grad():model.mode_embedding.weight.copy_(torch.as_tensor(d['mode_init'],device='cuda'))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0001)
    initial=tensor_state_digest(model.state_dict())
    frozen=tensor_state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('geometry.','head.feature_encoder.','head.state_encoder.'))})
    tensors={k:torch.as_tensor(d[k],device='cuda',dtype=torch.float32) for k in ('context','anchor','current','evidence','centers','halves','floors')}
    settings=dict(arm=arm,seed=seed,steps=steps,pair_weight=pair_weight,prepared_sha256=manifest['data_sha256'],initial_sha256=sha(INITIAL),
        source_commit=os.environ.get('CODE_COMMIT'),source_hashes={str(p.relative_to(SOURCE)):sha(p) for p in [SOURCE/'routeset/mode_geometry.py',SOURCE/'scripts/mode_geometry_experiment.py']},
        batch_size=32,lambda_mode=.001,lambda_pair=pair_weight,lr=.0003,encoder_frozen=True,independent_decoder=independent_decoder)
    history=[];start=0;stream='';tic=time.monotonic()
    def state(step):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng),step=step,history=history,
        stream=stream,settings=settings,config=dict(head_options=options,conditional=arm!='B',decoder_communication=not independent_decoder),initial_tensor_sha256=initial,frozen_sha256=frozen)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng)
        start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    end=min(steps,stop_after or steps)
    for step in range(start+1,end+1):
        chosen=rng.integers(len(d['pairs']),size=16);ids=d['pairs'][chosen].reshape(-1)
        modes=[];target=[];pairmask=[]
        for j in chosen:
            common=rng.permutation(np.flatnonzero(d['pair_mask'][j]))[:6]
            pairmask.append([1]*len(common)+[0]*(8-len(common)))
            for side,ident in enumerate(d['pairs'][j]):
                remaining=list(sorted(set(per_mode[ident])-set(common)))
                extra=rng.permutation(remaining)[:8-len(common)].tolist()
                while len(common)+len(extra)<8:extra.append(int(rng.choice(list(per_mode[ident]))))
                modes.append(np.r_[common,extra])
                paths=[d['pair_source' if side==0 else 'pair_target'][j,m] for m in common]
                paths.extend(s['paths'][ident,int(rng.choice(per_mode[ident][m]))] for m in extra)
                target.append(paths)
        modes=np.array(modes,dtype=np.int64);target=np.asarray(target,dtype=np.float32)
        stream=hashlib.sha256(stream.encode()+ids.tobytes()+modes.tobytes()+target.tobytes()).hexdigest()
        tx=torch.as_tensor(target,device='cuda');mi=torch.as_tensor(modes,device='cuda')
        xyz,event,details=model.decode(tensors['context'][ids],tensors['anchor'][ids],tensors['current'][ids],mi)
        if arm=='B':
            # Same positive paths, labels, pairs and displacement; only output parameterization differs.
            cost=(xyz[:,:,None]-tx[:,None]).square().mean((-1,-2)).detach().cpu().numpy()
            assignment=np.array([linear_sum_assignment(c.T)[1] for c in cost])
            xyz=xyz[torch.arange(32,device='cuda')[:,None],torch.as_tensor(assignment,device='cuda')]
            event=event[torch.arange(32,device='cuda')[:,None],torch.as_tensor(assignment,device='cuda')]
        route=(xyz[:,:,1:]-tx[:,:,1:]).square().mean()+.01*(event-tensors['current'][ids,None,None,7]).square().mean()
        clear=segment_clearance_loss(xyz,tensors['centers'][ids],tensors['halves'][ids])+workspace_floor_loss(xyz,tensors['floors'][ids])
        ml=mode_loss(details['mode_logits'],tensors['evidence'][ids])
        pl=pair_displacement_loss(xyz,tx,torch.as_tensor(pairmask,device='cuda',dtype=torch.float32))
        loss=route+160*clear+.001*ml+(pair_weight*pl if arm in ('B','D') else 0)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%100==0 or step==steps:
            row=dict(step=step,loss=float(loss),route=float(route),clear=float(clear),mode=float(ml),pair=float(pl))
            history.append(row);print(json.dumps(row),flush=True)
        if step%100==0 or step==end:atomic_checkpoint(out/'recovery.pt',state(step))
    assert frozen==tensor_state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('geometry.','head.feature_encoder.','head.state_encoder.'))})
    if end==steps:
        atomic_checkpoint(out/'last.pt',state(end))
        write(out/'summary.json',dict(steps=end,stream_sha256=stream,initial_tensor_sha256=initial,
            last_sha256=sha(out/'last.pt'),frozen_sha256=frozen,elapsed_seconds=time.monotonic()-tic,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))


def evaluate(name,sampling):
    torch=torch_setup()
    from routeset.mode_geometry import ModeGeometryHead,VOCAB
    from routeset.observed_probability import load_scored_planner
    from scripts.evaluate_paired_modes import inputs_for
    out=RUN/name/('eval_'+sampling);out.mkdir(exist_ok=False)
    saved=torch.load(RUN/name/'last.pt',map_location='cpu',weights_only=False)
    model=ModeGeometryHead(**saved['config']['head_options'],conditional=saved['config']['conditional'],decoder_communication=saved['config'].get('decoder_communication',True)).cuda()
    model.load_state_dict(saved['model']);model.eval();model.requires_grad_(False)
    assert sha(Q)=='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'
    scorer=load_scored_planner(Q,'cuda');scorer.eval();scorer.requires_grad_(False)
    torch.manual_seed(71239) # predeclared inference ablation seed; no pool expansion
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    rows=[r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'];assert len(rows)==288
    with np.load(old.SUPPORT/'support.npz') as z:known={str(k):set(z['modes'][i,z['mask'][i]]) for i,k in enumerate(z['ids']) if z['splits'][i]=='DEV_MODEL'}
    paths=[];events=[];qs=[];ids=[];hashes={};assign=[];logits=[];details=[]
    for row in rows:
        inp=inputs_for(row,labels[row['id']],DATA/'export/qwen_cache',torch,hashes)
        with torch.inference_mode():
            p,e,info=model(**inp,sampling=sampling);q=old.fixed_path_scores(scorer,p,e,inp)
        paths.append(p[0].cpu().numpy());events.append(e[0].cpu().numpy());qs.append(q[0].cpu().numpy());ids.append(row['id'])
        assign.append(info['mode_ids'][0].cpu().numpy());logits.append(info['mode_logits'][0].cpu().numpy())
    for i,row in enumerate(rows):
        ref=references(labels[row['id']]);_,cc=check_candidates(paths[i],events[i],ref['label'],ref['current'],ref['truth'],ref['config'])
        valid=np.array([c['TipValid'] for c in cc]);words=[mode(p,ref['config']) if v else None for p,v in zip(paths[i],valid)]
        chosen=select(paths[i],qs[i],4)
        details.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],variant=row['parent_id'].rsplit('_',1)[1],
            raw=coverage(words,valid,known[row['id']],list(range(8))),selected=coverage(words,valid,known[row['id']],chosen),
            words=words,valid=valid.tolist(),q=qs[i].tolist(),assigned_modes=[VOCAB[j] for j in assign[i]],candidates=cc,
            condition_hit=sum(v and w==VOCAB[j] for v,w,j in zip(valid,words,assign[i]))/8))
    np.savez_compressed(out/'pool.npz',paths=paths,events=events,q=qs,ids=ids,mode_ids=assign,mode_logits=logits)
    result=dict(raw=average([r['raw'] for r in details]),selected=average([r['selected'] for r in details]),
        condition_hit=float(np.mean([r['condition_hit'] for r in details])),generator_sha256=sha(RUN/name/'last.pt'),
        scorer_sha256=sha(Q),pool_sha256=sha(out/'pool.npz'),input_hashes=hashes,locked_access=False)
    write(out/'rows.json',plain(details));write(out/'metrics.json',plain(result));print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);p.add_argument('--arm',choices=['B','C','D'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=1200);p.add_argument('--name')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--pair-weight',type=float,default=1.)
    p.add_argument('--independent-decoder',action='store_true')
    p.add_argument('--sampling',choices=['ordinary','balanced','adaptive'],default='balanced');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':train(a.arm,a.seed,a.steps,a.name,a.resume,a.stop_after,a.pair_weight,a.independent_decoder)
    else:evaluate(a.name,a.sampling)
