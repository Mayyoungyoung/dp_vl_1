"""TRAIN-only one-shot spacing/stop association and native initial-state audit."""
import time
import numpy as np
from scipy.stats import rankdata
from research_selective_repair_v1.io import RUN,SOURCE,read,write,sha

def auc(value,positive):
    n=int(positive.sum());m=len(value)-n
    return float((rankdata(value)[positive].sum()-n*(n+1)/2)/(n*m)) if n and m else None

def run(name):
    tic=time.monotonic();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    file=RUN/'body_halfgoal_events_v2/samples.npz'
    with np.load(file) as z:d={k:z[k] for k in z.files}
    assert len(d['ids'])==1152 and set(d['roles'])=={'TRAIN'}
    delta=np.diff(d['paths'],axis=1);length=np.linalg.norm(delta,axis=-1)
    previous=np.concatenate([d['paths'][:,:1],d['prefix_tip'][:,:-1,-1]],1)
    actual_step=np.linalg.norm(d['paths'][:,1:]-previous,axis=-1)
    angle=np.zeros_like(length);angle[:,1:]=1-(delta[:,1:]*delta[:,:-1]).sum(-1)/np.maximum(length[:,1:]*length[:,:-1],1e-12)
    known=d['prefix_observed'];failed=d['prefix_hazard']>0;families=sorted(set(d['families']));held=np.isin(d['families'],families[-2:]);reports={}
    for label,mask in [('fit_TRAIN',~held),('held_TRAIN',held)]:
        selected=known&mask[:,None];positive=failed[selected];report={}
        for kind,value in [('nominal_step_m',length),('actual_endpoint_to_next_waypoint_m',actual_step),('turn_1minus_cos',angle)]:
            v=value[selected];report[kind]=dict(failure_median=float(np.median(v[positive])),completed_median=float(np.median(v[~positive])),stop_AUROC=auc(v,positive))
        report['observed_segments']=int(selected.sum());report['first_stops']=int(positive.sum());report['failure_segment_counts']=np.bincount(np.where(selected&failed)[1],minlength=23).tolist();reports[label]=report
    old=RUN/'body_execution_train_identity_families2_7_v1'
    baseline={read(p)['execution_id']:(p,read(p)) for p in old.glob('plan*.json')}
    comparisons=[]
    for repeat in range(2):
        folder=RUN/('body_zero_edit_TRAIN_repeat_suite_v1_repeat%d'%repeat)
        maximum={};first_divergence=[]
        for p in folder.glob('plan*.json'):
            plan=read(p);old_file,old_plan=baseline[plan['execution_id']]
            np.testing.assert_array_equal(plan['execution_paths'],old_plan['execution_paths']);assert plan['config']==old_plan['config'] and plan['seed']==old_plan['seed']
            fresh=folder/('parent'+p.stem[4:]);former=old/('parent'+old_file.stem[4:])
            a=read(former/'EXECUTION.json')['records'];b=read(fresh/'EXECUTION.json')['records']
            for ra,rb in zip(a,b):
                assert ra['slot']==rb['slot']
                fa=former/plan['parent_id']/ra['trace']['file'];fb=fresh/plan['parent_id']/rb['trace']['file']
                assert sha(fa)==ra['trace']['sha256'] and sha(fb)==rb['trace']['sha256']
                with np.load(fa) as za,np.load(fb) as zb:
                    for key in za.files:maximum[key]=max(maximum.get(key,0),float(np.max(np.abs(za[key][0]-zb[key][0]))))
                    n=min(len(za['arm_joint_positions']),len(zb['arm_joint_positions']))
                    differs=np.flatnonzero(np.max(np.abs(za['arm_joint_positions'][:n]-zb['arm_joint_positions'][:n]),axis=-1)>1e-8)
                    first_divergence.append(int(differs[0]) if len(differs) else None)
        comparisons.append(dict(repeat=repeat,maximum_initial_state_array_difference=maximum,first_joint_trace_divergence_step=first_divergence))
    result=dict(spacing=reports,zero_edit_initial_state=comparisons,seconds=time.monotonic()-tic,source_commit=SOURCE.name,
        dataset_sha256=sha(file),scope='TRAIN association,not causal spacing evidence; exact initial stored arrays do not certify hidden simulator/native RNG state',locked_access=False)
    write(out/'SUMMARY.json',result);print(result,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
