"""R0/R1/R2 paired input streams, observation-only forward, resumable state."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,PARENT,OLD,read,write,sha,lines,torch_setup,verify_role
from scripts.paired_modes_data import DATA,RUN,POLICY


def merge(data,geometry,extra,extra_geometry):
    assert not set(data['scene_ids'])&set(extra['scene_ids'])
    result=dict(data)
    for key in ('features','current','splits','scene_ids','parent_ids','image_hashes'):
        result[key]=np.concatenate([data[key],extra[key]])
    nref=max(data['paths'].shape[1],extra['paths'].shape[1])
    for key in ('paths','events','path_mask'):
        arrays=[]
        for d in (data,extra):
            a=d[key];pad=[(0,0)]*a.ndim;pad[1]=(0,nref-a.shape[1]);arrays.append(np.pad(a,pad))
        result[key]=np.concatenate(arrays)
    for key in ('semantic_targets','instructions','tasks','unreferenced','skipped'):result[key]=data[key]+extra[key]
    result['source_hashes']=dict(data['source_hashes'],**extra['source_hashes'])
    result['fingerprint']=hashlib.sha256(json.dumps(result['source_hashes'],sort_keys=True).encode()).hexdigest()
    g=dict(geometry);offset=len(g['points']['depth'])
    g['points']={k:np.concatenate([g['points'][k],extra_geometry['points'][k]]) for k in g['points']}
    g['index']=np.r_[g['index'],extra_geometry['index']+offset]
    g['fingerprint']=hashlib.sha256((geometry['fingerprint']+extra_geometry['fingerprint']).encode()).hexdigest()
    return result,g


def load_data():
    from routeset.observed_route_head import load_observed_dataset
    from scripts.train_observed_geometry import load_geometry
    pcfg=read(PARENT/'config.json');folders=[OLD,verify_role('FUTURE_GENERATOR_TRAIN'),DATA/'export']
    ds=geo=None;labelmap={}
    for folder in folders:
        d=load_observed_dataset(folder/'observations.jsonl',folder/'supervision.jsonl',folder/'qwen_cache',24,'both')
        g=load_geometry(d,folder/'observations.jsonl',folder/'supervision.jsonl',2)
        ds,geo=(d,g) if ds is None else merge(ds,geo,d,g)
        labelmap.update({r['id']:r for r in lines(folder/'supervision.jsonl')})
    configs=[];centers=[];halves=[];truth_hashes={}
    for ident in ds['scene_ids']:
        label=labelmap[str(ident)];c=read(label['route_config'])
        for key in ('route_config','verification_only'):truth_hashes[label[key]]=sha(label[key])
        if 'post_heights' not in c:c['post_heights']=[c['post_size_xyz'][2]]*len(c['row_x'])
        configs.append(c)
        with np.load(label['verification_only']) as z:
            centers.append(z['obstacle_centers']);halves.append(z['obstacle_halfsizes'])
    assert len({x.shape for x in centers})==1
    geo['fingerprint']=hashlib.sha256(json.dumps(dict(observed=geo['fingerprint'],supervision=truth_hashes),sort_keys=True).encode()).hexdigest()
    return ds,geo,configs,np.stack(centers),np.stack(halves),pcfg,labelmap


def paired_population(data):
    meta={r['id']:r for r in lines(DATA/'export/metadata.jsonl')};families={}
    old=[]
    for i,ident in enumerate(data['scene_ids']):
        if not data['path_mask'][i].any() or data['splits'][i] not in ('TRAIN','FUTURE_GENERATOR_TRAIN'):continue
        if ident in meta:
            m=meta[ident];assert m['role']=='TRAIN'
            key=(m['family_id'],ident.rsplit('_target',1)[1]);families.setdefault(key,[]).append(i)
        else:old.append(i)
    groups=[v for _,v in sorted(families.items()) if len(v)>=2]
    assert groups and len(old)==573
    return groups,np.asarray(old)


def sample_batch(rng,groups,old):
    ids=[];pairs=[]
    for g in rng.integers(len(groups),size=8):
        a,b=rng.choice(groups[int(g)],size=2,replace=False)
        pairs.append((len(ids),len(ids)+1));ids.extend([a,b])
    ids.extend(rng.choice(old,size=16));return np.asarray(ids,dtype=np.int64),pairs


def source_identity():
    return {str(p.relative_to(SOURCE)):sha(p) for directory in ('scripts','routeset','configs')
            for p in (SOURCE/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.sh')}


def train(arm,seed,steps=None,resume=False,probe=False,run_name=None,stop_after=None):
    torch=torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead,balanced_assignment_loss
    from routeset.paired_modes import class_weights,within_scene_loss,partial_pair_loss
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import segment_clearance_loss
    from routeset.train_v2 import positive_assignment_loss,atomic_checkpoint,rng_state,restore_rng
    from routeset.observed_training_audit import tensor_state_digest,append_indices,new_stream_audit
    from scripts.train_observed_geometry import batch_inputs
    from scripts.evaluate_observed_two_row_online import head_options
    from scripts.observed_layout_variation import crossing_signature
    policy=read(POLICY);pc=read(SOURCE/'configs/observed_probability_v1.json')['stage2']
    steps=steps or policy['steps'];name=run_name or ('probe' if probe else arm)+'_seed%d'%seed
    if Path(name).name!=name:raise ValueError('Run name must be a directory basename')
    out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed run cannot be continued')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    rng=np.random.default_rng(seed);loss_rng=np.random.default_rng(seed+9000)
    data,geometry,configs,cs,hs,pcfg,labelmap=load_data();groups,old=paired_population(data)
    ids_train=np.flatnonzero(np.isin(data['splits'],['TRAIN','FUTURE_GENERATOR_TRAIN'])&data['path_mask'].any(1))
    weights=np.zeros(data['path_mask'].shape,dtype=np.float32);modes=[[] for _ in configs]
    ww,mm=class_weights(data['paths'][ids_train],data['path_mask'][ids_train],[configs[i] for i in ids_train],crossing_signature)
    weights[ids_train]=ww
    for i,m in zip(ids_train,mm):modes[i]=m
    opts=head_options(pcfg);opts['max_candidates']=8
    model=ProbabilisticGeometryRouteHead(**opts).cuda()
    initpath=ROOT/'runs/segment_clearance_v1'/('B_seed%d'%seed)/'last.pt'
    assert sha(initpath)==read(initpath.parent/'summary.json')['last_checkpoint_sha256']
    model.load_state_dict(torch.load(initpath,map_location='cpu',weights_only=False)['model'])
    initial=tensor_state_digest(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=pc['lr'],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    settings=dict(arm=arm,seed=seed,steps=steps,policy=policy,parent_sha256=sha(initpath),lr=pc['lr'],
        data_fingerprint=geometry['fingerprint'],source_sha256=source_identity(),
        initial_sha256=initial,independent_input_sampler=True,old_per_step=16,new_per_step=16,
        head_options=opts)
    if resume:
        if read(out/'config.json')!=settings:raise ValueError('Saved run config differs')
    else:write(out/'config.json',settings)
    audit=new_stream_audit(model,rng.bit_generator.state,torch.get_rng_state());start_step=0;history=[]
    coefficients=read(RUN/'probe_seed0/coefficients.json') if not probe else None
    if resume:
        saved=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False)
        if saved['config']!=settings:raise ValueError('Resume config/source/data differs')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'],rng);loss_rng.bit_generator.state=saved['loss_rng'];start_step=saved['step'];audit=saved['sampler'];history=saved['history']
    cs=torch.as_tensor(cs,dtype=torch.float32,device='cuda');hs=torch.as_tensor(hs,dtype=torch.float32,device='cuda')
    tic=time.monotonic();ratios=[];cosines=[]
    def checkpoint(step):
        return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            rng=rng_state(rng),loss_rng=loss_rng.bit_generator.state,sampler=audit,config=settings,
            step=step,history=history,coefficients=coefficients,source_commit=os.environ.get('CODE_COMMIT'))
    end_step=4 if probe else min(steps,stop_after or steps)
    if start_step>=end_step:raise ValueError('No new steps requested')
    for step in range(start_step+1,end_step+1):
        ids,pairs=sample_batch(rng,groups,old);audit=append_indices(audit,ids)
        model.train();inp=batch_inputs(data,geometry,ids,'cuda');xyz,events,details=model(**inp)
        tx=torch.as_tensor(data['paths'][ids],device='cuda');te=torch.as_tensor(data['events'][ids],device='cuda')
        pred=torch.cat([xyz[:,:,1:],events[:,:,1:,None]*.2],-1);target=torch.cat([tx[:,:,1:],te[:,:,1:,None]*.2],-1)
        base=positive_assignment_loss(pred,target,data['path_mask'][ids],'saturation',loss_rng)
        ground=positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],tx[:,:,-1],
            torch.as_tensor(data['path_mask'][ids],device='cuda'),.025)
        clear=segment_clearance_loss(xyz,cs[ids],hs[ids]);ordinary=base+.02*ground+160*clear
        cfgs=[configs[i] for i in ids[:16]];types=[modes[i] for i in ids[:16]]
        rel=within_scene_loss(xyz[:16],cfgs,types)
        pair,matched=partial_pair_loss(xyz[:16],cfgs,types,pairs)
        if probe:
            params=[p for p in model.parameters() if p.requires_grad]
            norms=[];grads=[]
            for term in (ordinary,rel,pair):
                grad=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
                norms.append(float(torch.sqrt(sum(g.square().sum() for g in grad if g is not None))))
                grads.append(grad)
            cosines.append([float(sum((a*b).sum() for a,b in zip(grads[0],g) if a is not None and b is not None))/(norms[0]*n+1e-30)
                            for g,n in zip(grads[1:],norms[1:])])
            ratios.append(norms);continue
        # Relation-balanced regression only on new paired scenes; same in R1/R2.
        if arm!='R0':
            balanced,_=balanced_assignment_loss(pred[:16],target[:16],torch.as_tensor(weights[ids[:16]],device='cuda'))
            # The old half keeps exactly its original target objective.
            old_loss=positive_assignment_loss(pred[16:],target[16:],data['path_mask'][ids[16:]],'saturation',loss_rng)
            loss=.5*(balanced+old_loss)+.02*ground+160*clear+coefficients['relation']*rel
            if arm=='R2':loss=loss+coefficients['pair']*pair
        else:loss=ordinary
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step()
        if step%100==0 or step==steps:
            row=dict(step=step,loss=float(loss),clearance=float(clear),relation=float(rel),pair=float(pair),matched=matched)
            history.append(row);print(json.dumps(row),flush=True)
        if step%policy['checkpoint_every']==0 or step==end_step:atomic_checkpoint(out/'recovery.pt',checkpoint(step))
    if probe:
        a=np.asarray(ratios)
        if not np.isfinite(a).all() or np.any(a<=1e-10):raise ValueError('Degenerate TRAIN gradient probe')
        coefficients=dict(relation=float(np.median(.2*a[:,0]/np.maximum(a[:,1],1e-12))),
                          pair=float(np.median(.2*a[:,0]/np.maximum(a[:,2],1e-12))),
                          rule='0.2 * original total parameter-gradient norm / added-loss norm; four fixed TRAIN batches; no updates',
                          gradient_norms=ratios,cosine_with_ordinary=cosines)
        write(out/'coefficients.json',coefficients);return
    if end_step<steps:
        write(out/'interruption.json',dict(step=end_step,planned_steps=steps,reason='Controlled resume verification',
            elapsed_seconds=time.monotonic()-tic,recovery_sha256=sha(out/'recovery.pt')))
        return
    atomic_checkpoint(out/'last.pt',checkpoint(steps))
    final=tensor_state_digest(model.state_dict());assert initial!=final
    write(out/'summary.json',dict(arm=arm,seed=seed,steps=steps,elapsed_seconds=time.monotonic()-tic,
        initial_sha256=initial,final_sha256=final,last_sha256=sha(out/'last.pt'),sampler=audit,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),history=history))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['R0','R1','R2'],default='R0');p.add_argument('--run-name');p.add_argument('--stop-after',type=int)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--probe',action='store_true')
    a=p.parse_args();train(a.arm,a.seed,a.steps,a.resume,a.probe,a.run_name,a.stop_after)
