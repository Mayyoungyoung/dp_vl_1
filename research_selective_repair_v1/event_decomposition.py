"""TRAIN-only counterfactual composition audit; oracle arms never deployment."""
import argparse,copy,time
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,SOURCE,read,write,sha
from research_selective_repair_v1.body_event_forecast import compose,success

def metrics(prob,labels,mask):
    selected=prob[mask];actual=labels[mask];positive=actual>0
    conditional=selected[:,1:]/np.maximum(selected[:,1:].sum(-1,keepdims=True),1e-12)
    return dict(rows=int(mask.sum()),positive_rows=int(positive.sum()),
        full_composed_successful_clear_brier=float(np.square(1-selected[:,0]-positive).mean()),
        conditional_word_accuracy=float((conditional.argmax(-1)[positive]+1==actual[positive]).mean()),
        conditional_word_nll=float(-np.log(np.maximum(conditional[np.flatnonzero(positive),actual[positive]-1],1e-12)).mean()),
        positive_mean_word_mass=float(selected[np.flatnonzero(positive),actual[positive]].mean()))

def run(name):
    tic=time.monotonic();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    dataset=RUN/'body_halfgoal_events_v2'/'samples.npz'
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'} and len(d['ids'])==1152
    registry=read(ROOT/'data/selective_repair_interventions_v1/registration.json')
    plans={p['parent_id']:p for p in registry['parent_plan'] if p['role']=='TRAIN'}
    settings=read(RUN/'constraints_seed0_v1'/'config.json');base=settings['post_base']
    records=[];hashes={str(dataset):sha(dataset)}
    for kind in ('recurrent','nonrecurrent'):
        prediction=RUN/('body_event_pilot_%s_seed0_v2'%kind)/'TRAIN_diagnostic_predictions.npz'
        hashes[str(prediction)]=sha(prediction)
        with np.load(prediction) as z:
            assert np.array_equal(z['ids'],d['ids']) and np.array_equal(z['slots'],d['slots']) and np.array_equal(z['options'],d['options'])
            p={k:z[k] for k in ('mean','scale','hazard','events','log_weights','clear')};held=z['heldout']
        for geometry in ('current_NN','oracle_TRAIN_config'):
            for signal in ('predicted','current_requested_crossing_timing','oracle_TRAIN_timing','oracle_TRAIN_timing_and_positions'):
                q=copy.deepcopy(p)
                if signal.startswith('oracle_'):q['events']=np.broadcast_to(np.where(d['event_present'][:,None],40.,-40.),p['events'].shape).copy()
                if signal=='current_requested_crossing_timing':
                    rowx=d['completed'].reshape(-1,2,2,3)[...,0].mean(2)
                    hit=(d['paths'][:,:-1,None,0]<rowx[:,None])&(d['paths'][:,1:,None,0]>=rowx[:,None])
                    index=hit.argmax(1);present=np.zeros_like(d['event_present'])
                    for row in range(2):present[np.arange(len(present)),index[:,row],row]=hit[:,:,row].any(1)
                    q['events']=np.broadcast_to(np.where(present[:,None],40.,-40.),p['events'].shape).copy()
                if signal=='oracle_TRAIN_timing_and_positions':
                    q['mean']=np.broadcast_to(d['event_tip'][:,None,...,1:],p['mean'].shape).copy()
                    q['scale']=np.full_like(p['scale'],1e-5)
                prob=np.empty((len(d['ids']),17),np.float32)
                for ident in sorted(set(d['ids'])):
                    ix=np.flatnonzero(d['ids']==ident)
                    if geometry=='oracle_TRAIN_config':cfg=plans[str(ident).rsplit('_target',1)[0]]['config']
                    else:
                        completed=d['completed'][ix[0]];height=np.clip(completed[:,2]-base,.025,.4)
                        cfg=dict(row_x=[float(completed[j:j+2,0].mean()) for j in (0,2)],post_y=[completed[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(height[j:j+2].max()) for j in (0,2)],post_base_z=base,tip_clearance_m=.02)
                    prob[ix]=compose({k:v[ix] for k,v in q.items()},d['paths'][ix],cfg)[0]
                records.append(dict(kind=kind,geometry=geometry,event_signal=signal,fit=metrics(prob,d['labels'],~held),held_TRAIN=metrics(prob,d['labels'],held)))
        proxy=success(p)
        records.append(dict(kind=kind,proxy_only_brier_held_TRAIN=float(np.square(proxy[held]-(d['labels'][held]>0)).mean()),warning='Proxy omits spatial word composition'))
    write(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-tic,source_commit=SOURCE.name,input_sha256=hashes,
        scope='1152 TRAIN diagnostic rows only, held TRAIN families6/7; oracle geometry/timing/positions counterfactuals isolate errors, never method inputs or deployment results',locked_access=False))
    print(records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
