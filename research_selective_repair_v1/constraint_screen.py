"""Seal observed predictions before independent DEV labels or metrics are read."""
import argparse,time
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.constraints import load,summaries,boxes
from research_selective_repair_v1.constraint_operator import repair
from routeset.observed_probability import load_scored_planner,route_observation_features


def screen(name,checkpoint,data_name,data,kind='protected',batch=16,role='DEV_MODEL'):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    model,ck=load(checkpoint)
    # Legal observation intermediates only. No target/valid/truth arrays loaded here.
    assert role in ('TRAIN','DEV_MODEL')
    with np.load(RUN/data_name/'samples.npz') as z:
        selected_role=z['splits']==role;d={k:z[k][selected_role] for k in ('ids','families','drafts','events','context','modes','local','visible_points','visible_mask')}
    x,anchor=summaries(d['visible_points'],d['visible_mask'],np.asarray(ck['settings']['priors']))
    t=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    with torch.no_grad():completion=model(t(d['context']),t(x),t(anchor));centers,halves=boxes(completion,ck['settings'])
    paths=[];supports=[];steps=[];tic=time.monotonic()
    goal=d['drafts'][:,:,-1]+.4*d['local'][:,:,0,29:32]
    assert np.max(np.ptp(goal,axis=1))<1e-5
    for i in range(0,len(x),batch):
        j=min(i+batch,len(x))
        p,info=repair(t(d['drafts'][i:j]),torch.tensor(d['modes'][i:j],device='cuda'),t(goal[i:j,0]),t(d['local'][i:j,0,0,32]),centers[i:j],halves[i:j],ck['settings'],kind)
        paths.extend(p.cpu().numpy());supports.extend(info['support'].cpu().numpy());steps.append(info['steps'])
    paths=np.asarray(paths);np.savez_compressed(out/'sealed_predictions.npz',ids=d['ids'],paths=paths,events=d['events'],modes=d['modes'],drafts=d['drafts'],support=supports,completed=completion.cpu().numpy())
    prediction_seconds=time.monotonic()-tic
    seal=sha(out/'sealed_predictions.npz');write(out/'SEAL.json',dict(prediction_sha256=seal,checkpoint_sha256=sha(checkpoint),current_observation_only=True,operator_steps=64,batch_steps=steps,kind=kind,role=role))
    # Independent truth/checker starts after the prediction seal above.
    with np.load(RUN/data_name/'samples.npz') as z:
        truth=np.concatenate([z['obstacle_centers'][selected_role,...,:2],(z['obstacle_centers'][selected_role,...,2]+z['obstacle_halves'][selected_role,...,2])[...,None]],-1)
    observations={r['id']:r for r in lines(data/'export/observations.jsonl') if r['split']==role}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']==role}
    scorer=load_scored_planner(Q,'cuda').requires_grad_(False);center,_=center_model();values=[];qs=[];valids=[];selected=[];hashes={}
    for i,ident in enumerate(d['ids']):
        ident=str(ident);inp=inputs_for(observations[ident],labels[ident],data/'export/qwen_cache',torch,hashes);runner=SceneRunner(center,scorer,inp)
        with torch.no_grad():
            p=t(paths[i:i+1]);ev=t(d['events'][i:i+1]);nodes,ctx=route_observation_features(p,ev,inp['current'],inp['world_xyz'],inp['rgb'],inp['valid_mask'],runner.qgeo['point_features'],runner.qcontext,runner.qgeo['anchor_xyz'])
            q=(scorer.scorer((nodes-scorer.nodes_mean)/scorer.nodes_std,(ctx-scorer.context_mean)/scorer.context_std)/scorer.temperature).sigmoid().cpu().numpy()
        ref=references(labels[ident]);a=assess(paths[i:i+1],d['events'][i:i+1],q,ref);b=assess(d['drafts'][i:i+1],d['events'][i:i+1],q,ref)
        old=wordset(d['drafts'][i],b['valid'][0],ref['config']);new=wordset(paths[i],a['valid'][0],ref['config'])
        values.append(dict(id=ident,family=str(d['families'][i]),utility=a['utility'][0].tolist(),added=len(new-old),lost=len(old-new),repaired=int((~b['valid'][0]&a['valid'][0]).sum()),damaged=int((b['valid'][0]&~a['valid'][0]).sum()),base_valid=int(b['valid'][0].sum()),raw_mode_changed=int((a['raw_words'][0]!=b['raw_words'][0]).sum())))
        qs.append(q[0]);valids.append(a['valid'][0]);selected.append(a['selected'][0])
    assert sha(out/'sealed_predictions.npz')==seal
    np.savez_compressed(out/'pool.npz',ids=d['ids'],paths=paths,events=d['events'],drafts=d['drafts'],modes=d['modes'],q=qs,valid=valids,selected=selected)
    error=np.abs(completion.cpu().numpy()-truth)
    report=dict(kind=kind,checkpoint_sha256=sha(checkpoint),mean_utility=np.mean([v['utility'] for v in values],axis=0).tolist(),repaired=sum(v['repaired'] for v in values),damaged=sum(v['damaged'] for v in values),added=sum(v['added'] for v in values),lost=sum(v['lost'] for v in values),raw_mode_changed=sum(v['raw_mode_changed'] for v in values),damage_fraction=sum(v['damaged'] for v in values)/sum(v['base_valid'] for v in values),completion_mae_xyz=error.mean((0,1)).tolist(),completion_p95_max_m=float(np.quantile(error.max(-1),.95)),prediction_seconds=prediction_seconds,requests=len(x),steps_per_active_batch=64,head_scope='Screening frozen C0 success head; own matched TRAIN heads required before acceptance',locked_access=False)
    report['role']=role
    write(out/'ROWS.json',values);write(out/'SUMMARY.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-name',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--kind',choices=['protected','global','mode_global'],default='protected');p.add_argument('--batch',type=int,default=16);p.add_argument('--role',choices=['TRAIN','DEV_MODEL'],default='DEV_MODEL');screen(**vars(p.parse_args()))
