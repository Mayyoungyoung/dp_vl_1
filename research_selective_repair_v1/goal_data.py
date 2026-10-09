"""Measured minimal pivot: current-observation goal-tail correction targets."""
import argparse,time,os
import numpy as np
from PIL import Image
from research_selective_repair_v1.core import *
from scripts import observation_prototype_grounding as proto
from research_selective_repair_v1.grounding import predict
from scripts.research_v3_audit import mode

def build(name='goal_prepared_v1',source='prepared_v1',data=None,prototype=None):
    data=data or DATA;out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    rows=lines(data/'export/observations.jsonl');labels={r['id']:r for r in lines(data/'export/supervision.jsonl')}
    if prototype:model=read(prototype)
    else:
        model,_=proto.fit_prototypes(data/'export',rows,labels);write(out/'PROTOTYPES.json',model)
    with np.load(RUN/source/'samples.npz') as z:d={k:z[k] for k in z.files}
    rb={r['id']:r for r in rows};proposal=[];targets=d['targets'].copy();usable=d['usable'].copy();found=0;new_words={};statuses=[]
    for i,ident in enumerate(d['ids']):
        ident=str(ident);row=rb[ident];label=labels[ident]
        (rgb,xyz,valid),_=proto.load_observation(data/'export',row,label['observation'])
        endpoint,details=predict(rgb,xyz,valid,row['instruction'],model)
        if endpoint is None:endpoint=d['anchor'][i];available=0
        else:available=1
        # Same proposed target evidence goes to every learned and geometric arm.
        endpoint_features=np.concatenate([np.broadcast_to((endpoint-d['drafts'][i,:,-1])/ .4,(24,8,3)).transpose(1,0,2),np.full((8,24,1),available)],-1)
        proposal.append(endpoint_features);statuses.append(dict(id=ident,available=available,details=details,endpoint=np.asarray(endpoint).tolist()))
        ref=references(label);goal=np.asarray(label['semantic_targets']['centers'])[label['semantic_targets']['target_index']]
        bad=np.linalg.norm(d['drafts'][i,:,-1]-goal,axis=-1)>.03
        oldwords=[mode(p,ref['config']) for p in d['drafts'][i]]
        before=wordset(d['drafts'][i],d['valid'][i],ref['config'])
        for k in np.flatnonzero(bad):
            candidate=d['drafts'][i,k].copy();s=np.linspace(0,1,7)[1:];s=s*s*(3-2*s)
            candidate[18:]+=s[:,None]*(goal-candidate[-1])[None]
            _,check=check_candidates(candidate[None],d['events'][i,k:k+1],ref['label'],ref['current'],ref['truth'],ref['config'])
            w=mode(candidate,ref['config'])
            if check[0]['TipValid'] and (oldwords[k] is None or oldwords[k]==w):targets[i,k]=candidate;usable[i,k]=True;found+=1
        after=wordset(targets[i],usable[i],ref['config']);role=str(d['splits'][i]);new_words[role]=new_words.get(role,0)+len(after-before)
    d['local']=np.concatenate([d['local'],np.asarray(proposal,dtype=np.float32)],-1)
    d['targets']=targets;d['usable']=usable;d['support']=(np.linalg.norm(targets-d['drafts'],axis=-1)>.001).astype(np.float32)
    np.savez_compressed(out/'samples.npz',**d);write(out/'OBSERVED_GOAL_PROPOSALS.json',statuses)
    write(out/'MANIFEST.json',dict(source_samples_sha256=sha(RUN/source/'samples.npz'),samples_sha256=sha(out/'samples.npz'),teacher_goal_repaired_routes=found,found_added_words=new_words,prototype_sha256=sha(prototype or out/'PROTOTYPES.json'),prototype_scope='TRAIN positive endpoints and observed color/extent components; no DEV fitting',source_commit=os.environ.get('CODE_COMMIT'),seconds=time.monotonic()-tic,locked_access=False,scope='Goal-tail correction pivot; same-mode claims only for defined actual original word. Unclassified originals count new validity, not same-mode survival.'))

def recover(name='goal_prepared_v1',source='prepared_v1'):
    """Recover only the completed arrays from the recorded final-metadata failure."""
    out=RUN/name
    assert not (out/'MANIFEST.json').exists()
    with np.load(out/'samples.npz') as z:d={k:z[k] for k in z.files}
    with np.load(RUN/source/'samples.npz') as z:old={k:z[k] for k in ('ids','targets','splits')}
    assert len(d['ids'])==1440 and np.array_equal(d['ids'],old['ids']) and d['local'].shape==(1440,8,24,33)
    assert np.array_equal(d['splits'],old['splits']) and all(np.isfinite(d[k]).all() for k in ('local','targets','drafts'))
    changed=(np.linalg.norm(d['targets']-old['targets'],axis=-1)>1e-6).any(-1)
    write(out/'MANIFEST.json',dict(source_samples_sha256=sha(RUN/source/'samples.npz'),samples_sha256=sha(out/'samples.npz'),prototype_sha256=sha(out/'PROTOTYPES.json'),teacher_goal_repaired_routes=int(changed.sum()),prototype_scope='TRAIN positive endpoints only',source_commit=os.environ.get('CODE_COMMIT'),recovered_from='goal_prepare_v1 NameError(os) only at final manifest after samples/prototypes/proposals completed',original_receipt_status='failed preserved',locked_access=False,arrays_complete=True))
    print(read(out/'MANIFEST.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='goal_prepared_v1');p.add_argument('--source',default='prepared_v1');p.add_argument('--data',type=Path);p.add_argument('--prototype',type=Path);build(**vars(p.parse_args()))
