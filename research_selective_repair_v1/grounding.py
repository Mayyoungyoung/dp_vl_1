"""Shared TRAIN-fitted observed-component workspace and surface-bias control."""
import argparse,copy
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,DATA,read,write,sha,lines
from scripts import observation_prototype_grounding as proto

def predict(rgb,xyz,valid,instruction,model):
    prior=model.get('spatial_prior')
    if prior is None:return proto.predict(rgb,xyz,valid,instruction,model)
    lower=np.asarray(prior['lower']);upper=np.asarray(prior['upper'])
    eligible=valid&((xyz>=lower)&(xyz<=upper)).all(-1)
    endpoint,info=proto.predict(rgb,xyz,eligible,instruction,model)
    if endpoint is not None:
        endpoint=endpoint+np.asarray(model.get('surface_offsets',{}).get(instruction,[0.,0.,0.]))
    return endpoint,dict(**info,workspace_rule='TRAIN target envelope plus TRAIN extent margin',workspace_visible_points=int(eligible.sum()))

def fit(name,parent,data=None):
    data=data or DATA;out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in lines(data/'export/observations.jsonl') if r['split']=='TRAIN']
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']=='TRAIN'}
    model=copy.deepcopy(read(parent));goals=[]
    for row in rows:
        target=labels[row['id']]['semantic_targets'];goals.append(target['centers'][target['target_index']])
    goals=np.asarray(goals);margin=max(.04,float(np.median([v['extent_m'] for v in model['prototypes'].values()])))
    model['spatial_prior']=dict(lower=(goals.min(0)-margin).tolist(),upper=(goals.max(0)+margin).tolist(),margin_m=margin,TRAIN_requests=len(rows),source='TRAIN semantic targets only; no DEV fitting')
    offsets={};records=[]
    for row,goal in zip(rows,goals):
        label=labels[row['id']];(rgb,xyz,valid),_=proto.load_observation(data/'export',row,label['observation'])
        point,info=predict(rgb,xyz,valid,row['instruction'],model)
        accepted=point is not None and np.linalg.norm(point-goal)<=.03
        if accepted:offsets.setdefault(row['instruction'],[]).append(goal-point)
        records.append(dict(id=row['id'],used=bool(accepted),observed_status=info['status']))
    model['surface_offsets']={k:np.median(v,axis=0).tolist() for k,v in offsets.items()}
    model['spatial_fit']=dict(parent_sha256=sha(parent),TRAIN_records=records,offset_counts={k:len(v) for k,v in offsets.items()},supervision_sha256=sha(data/'export/supervision.jsonl'),scope='Filter current observed points by TRAIN envelope; median TRAIN residual calibrates visible surface bias; labels never enter predict')
    write(out/'PROTOTYPES.json',model);print(dict(name=name,prior=model['spatial_prior'],offsets=model['surface_offsets'],counts=model['spatial_fit']['offset_counts']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--data',type=Path);fit(**vars(p.parse_args()))
