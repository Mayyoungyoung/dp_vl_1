"""Ordinary M8 paired continuation; only extra TRAIN segment-clearance term.

Loop copied from train_observed_probability_set at8faf122; historical A remains immutable.
"""
import argparse
import hashlib
import json
import random
import time

import numpy as np
from scripts.run_observed_probability import ROOT,RUN,OLD,PARENT,POLICY,sha,read,write,torch_setup,verify_role


def evaluate_composite(*args,**kwargs):
    from scripts import train_observed_two_row as ordinary
    from scripts.export_two_row_composite_observations import verify_export
    from routeset.observed_qwen_continuation import composite_evaluator
    with composite_evaluator(ordinary,verify_export):
        return ordinary.evaluate(*args,**kwargs)


def append_generator_data(data,geometry,extra,extra_geometry):
    if set(extra['splits'])!={'FUTURE_GENERATOR_TRAIN'}:
        raise ValueError('Generator expansion accepts only its prospectively reserved TRAIN role')
    if set(data['parent_ids']) & set(extra['parent_ids']):raise ValueError('Generator parent overlap')
    result=dict(data)
    for key in ('features','current','splits','scene_ids','parent_ids','image_hashes'):
        result[key]=np.concatenate([data[key],extra[key]])
    r=max(data['paths'].shape[1],extra['paths'].shape[1])
    for key in ('paths','events','path_mask'):
        arrays=[]
        for d in (data,extra):
            a=d[key];pad=[(0,0)]*a.ndim;pad[1]=(0,r-a.shape[1]);arrays.append(np.pad(a,pad))
        result[key]=np.concatenate(arrays)
    for key in ('semantic_targets','instructions','tasks','unreferenced','skipped'):
        result[key]=data[key]+extra[key]
    result['source_hashes']=dict(data['source_hashes'],**extra['source_hashes'])
    result['fingerprint']=hashlib.sha256(json.dumps(result['source_hashes'],sort_keys=True).encode()).hexdigest()
    combined=dict(geometry)
    n=len(geometry['points']['depth'])
    combined['points']={k:np.concatenate([geometry['points'][k],extra_geometry['points'][k]]) for k in geometry['points']}
    combined['index']=np.concatenate([geometry['index'],extra_geometry['index']+n])
    combined['fingerprint']=hashlib.sha256((geometry['fingerprint']+extra_geometry['fingerprint']).encode()).hexdigest()
    return result,combined


def train(arm,seed,expanded_data=True,probe=False):
    torch=torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead,reference_cluster_weights,balanced_assignment_loss
    from routeset.observed_route_head import load_observed_dataset
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.train_v2 import positive_assignment_loss,atomic_checkpoint,rng_state
    from scripts.train_observed_geometry import load_geometry,batch_inputs
    from scripts.evaluate_observed_two_row_online import head_options
    from scripts.export_two_row_composite_observations import verify_export
    cfg=read(POLICY)['stage2']
    output=ROOT/'runs/segment_clearance_v1'/('probe' if probe else 'B_seed%d'%seed);output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    pcfg=read(PARENT/'config.json')
    verify_export(OLD)
    if sha(PARENT/'last.pt')!=read(POLICY)['parent_checkpoint_sha256']:raise ValueError('Parent changed')
    data=load_observed_dataset(pcfg['observations'],pcfg['supervision'],pcfg['cache_dir'],24,'both')
    geometry=load_geometry(data,pcfg['observations'],pcfg['supervision'],2)
    if geometry['fingerprint']!=pcfg['dataset_fingerprint']:raise ValueError('Original dataset changed')
    if expanded_data:
        path=verify_role('FUTURE_GENERATOR_TRAIN')
        extra=load_observed_dataset(path/'observations.jsonl',path/'supervision.jsonl',path/'qwen_cache',24,'both')
        extra_geometry=load_geometry(extra,path/'observations.jsonl',path/'supervision.jsonl',2)
        data,geometry=append_generator_data(data,geometry,extra,extra_geometry)
    train_ids=np.flatnonzero(np.isin(data['splits'],['TRAIN','FUTURE_GENERATOR_TRAIN']) & data['path_mask'].any(1));dev_ids=np.flatnonzero(data['splits']=='DEV_MODEL')
    if len(train_ids)!=(573 if expanded_data else 285) or len(dev_ids)!=36:raise ValueError('Fixed registered train/dev populations required')
    opts=head_options(pcfg);opts['max_candidates']=8
    model=ProbabilisticGeometryRouteHead(**opts).cuda()
    parent=torch.load(PARENT/'last.pt',map_location='cpu',weights_only=False)['model']
    expanded=model.state_dict()
    for key,value in parent.items():
        if key=='head.queries':
            expanded[key][:4].copy_(value)
            # Additional full-route queries start near the original queries,
            # with a seeded perturbation; same exact init in both paired arms.
            expanded[key][4:].copy_(value+torch.randn_like(value)*.1)
        else:expanded[key].copy_(value)
    model.load_state_dict(expanded)
    from routeset.observed_training_audit import tensor_state_digest
    initial=tensor_state_digest(model.state_dict())
    # Exclude dev references from both loss construction and grouping.
    weights=np.zeros(data['path_mask'].shape,dtype='float32')
    weights[train_ids]=reference_cluster_weights(data['paths'][train_ids],data['path_mask'][train_ids],cfg['reference_cluster_distance_m'])
    from scripts.segment_clearance_support import training_boxes, audit_pair, probe_loss
    centers,halves,box_hashes=training_boxes(data,train_ids,pcfg)
    pair=audit_pair(seed,initial,geometry,train_ids,data,cfg)
    write(output/'pair_audit.json',pair)
    write(output/'training_geometry_sha256.json',box_hashes)
    if probe:
        probe_loss(model,data,geometry,train_ids,centers,halves,rng,output)
        return
    clearance_cfg=read(__import__('pathlib').Path(__file__).resolve().parents[1]/'configs/segment_clearance_v1.json')
    coefficient=clearance_cfg['lambda']
    assert coefficient is not None and coefficient>0
    from routeset.segment_clearance import segment_clearance_loss
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.0001)
    history=[];drawhash=hashlib.sha256();started=time.perf_counter();best=-float('inf')
    write(output/'initialization.json',dict(model_sha256=initial,parent_sha256=sha(PARENT/'last.pt'),seed=seed,
        optimizer='fresh AdamW; weight transfer, not resumed optimization',extra_queries='seeded parent-query perturbations',
        train_parent_ids=sorted(set(map(str,data['parent_ids'][train_ids]))),reference_weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest()))
    for step in range(1,cfg['steps']+1):
        model.train();ids=rng.choice(train_ids,size=cfg['batch_size']);drawhash.update(ids.astype('<i8').tobytes())
        inputs=batch_inputs(data,geometry,ids,'cuda');xyz,events,details=model(**inputs)
        target_xyz=torch.tensor(data['paths'][ids],device='cuda');target_events=torch.tensor(data['events'][ids],device='cuda')
        prediction=torch.cat([xyz[:,:,1:],events[:,:,1:,None]*.2],-1)
        target=torch.cat([target_xyz[:,:,1:],target_events[:,:,1:,None]*.2],-1)
        w=torch.tensor(weights[ids],device='cuda')
        if arm in ('ordinary','clearance'):
            route_loss=positive_assignment_loss(prediction,target,data['path_mask'][ids],'saturation',rng)
            pi_loss=xyz.new_zeros(())
        else:
            route_loss,mass=balanced_assignment_loss(prediction,target,w)
            pi_loss=-(mass*details['pi_logits'].log_softmax(-1)).sum(1).mean()
        grounding=positive_endpoint_attention_loss(details['attention'],inputs['world_xyz'],inputs['valid_mask'],
            target_xyz[:,:,-1],torch.tensor(data['path_mask'][ids],device='cuda'),.025)
        original_loss=route_loss+.02*grounding+cfg['pi_loss_weight']*pi_loss
        clearance_loss=segment_clearance_loss(xyz,centers[ids],halves[ids])
        loss=original_loss+coefficient*clearance_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite generator loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss),route_loss=float(route_loss),pi_loss=float(pi_loss),clearance_loss=float(clearance_loss),lambda_clearance=coefficient)),flush=True)
        if step%100==0:
            atomic_checkpoint(output/'recovery.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng),
                step=step,config=cfg,arm=arm,seed=seed,history=history,draw_sha256=drawhash.hexdigest(),initial_sha256=initial,
                parent_sha256=sha(PARENT/'last.pt'),dataset_fingerprint=geometry['fingerprint'],elapsed_seconds=time.perf_counter()-started))
        if step%500==0:
            # Evaluate fixed development requests, preserving the training RNG.
            saved_rng=torch.get_rng_state();saved_cuda=torch.cuda.get_rng_state_all()
            metrics=evaluate_composite(model,data,geometry,dev_ids,'cuda',output/'dev'/('step%05d'%step),
                selection_metric='tip_unique_valid',evaluation_sources=(pcfg['observations'],pcfg['supervision']))
            torch.set_rng_state(saved_rng);torch.cuda.set_rng_state_all(saved_cuda)
            score=metrics['selection_score'];history.append(dict(step=step,metrics=metrics))
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng),step=step,config=cfg,
                arm=arm,seed=seed,history=history,draw_sha256=drawhash.hexdigest(),initial_sha256=initial,
                parent_sha256=sha(PARENT/'last.pt'),dataset_fingerprint=geometry['fingerprint'],elapsed_seconds=time.perf_counter()-started)
            atomic_checkpoint(output/'last.pt',state)
            if score>best:best=score;atomic_checkpoint(output/'best.pt',state)
    assert drawhash.hexdigest()==pair['historical_draw_sha256'], 'Paired sampler mismatch'
    write(output/'summary.json',dict(arm=arm,seed=seed,initial_sha256=initial,draw_sha256=drawhash.hexdigest(),
        fixed_last_metrics=history[-1]['metrics'],history=history,steps=cfg['steps'],input_draws=cfg['steps']*cfg['batch_size'],
        training_path_states=cfg['steps']*cfg['batch_size']*8,dev_requests=len(history)*36,dev_path_states=len(history)*36*8,
        last_checkpoint_sha256=sha(output/'last.pt'),best_checkpoint_sha256=sha(output/'best.pt'),
        elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        train_parents=len(set(data['parent_ids'][train_ids])),train_inputs=len(train_ids),expanded_data=expanded_data,
        scientific_scope='M8 paired same-data arms; pi+balanced-assignment bundle, not isolated pi causality; old DEV reused'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--probe',action='store_true')
    a=p.parse_args();train('clearance',a.seed,True,a.probe)
