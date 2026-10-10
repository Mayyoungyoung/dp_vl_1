"""Dense successful-prefix states and observed first-failure TRAIN supervision."""
import argparse
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_feedback_data import DEFAULT
from research_selective_repair_v1.public_kinematics import load,forward

def prefix(record,joints,tips):
    # Four normalized samples of each COMPLETED segment. A failed segment's
    # unfinished trace is not falsely labeled as its successful endpoint.
    q=np.zeros((23,4,7),np.float32);p=np.zeros((23,4,3),np.float32)
    valid=np.zeros(23,bool);hazard=np.zeros(23,np.float32);observed=np.zeros(23,bool)
    cursor=0
    for segment in record['planning_segments']:
        j=int(segment['segment']);n=int(segment['simulated_steps']);assert 0<=j<23
        observed[j]=True;success=segment['simulation_status']=='success'
        hazard[j]=not success
        if success:
            assert n>0
            indices=cursor+np.maximum(1,np.ceil(np.arange(1,5)*n/4).astype(int))
            q[j]=joints[indices];p[j]=tips[indices];valid[j]=True
        cursor+=n
        if not success:break
    assert cursor==len(joints)-1,'Every observed simulator step must be accounted for'
    return q,p,valid,hazard,observed

def build(name,dataset,model,expanded=False):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    lookup={(str(i),int(s),int(o)):j for j,(i,s,o) in enumerate(zip(d['ids'],d['slots'],d['options']))}
    sources=list(DEFAULT)
    if expanded:
        sources += [('body_execution_train_%s_%s_v3'%(n,scope),j,'semantics_v2')
            for scope in ('families0_7_targets12','families8_15_targets012')
            for j,n in enumerate(('identity','lift','preserved'))]
    rows={};hashes={};qref,base,relative=load(model);all_errors=[]
    for source,option,_ in sources:
        folder=RUN/source;manifest=read(folder/'MANIFEST.json')
        assert manifest['TRAIN_feedback_only'] and not manifest['locked_access']
        assert (folder/'SUMMARY.json').exists()
        hashes[str(folder/'MANIFEST.json')]=sha(folder/'MANIFEST.json')
        for file in sorted(folder.glob('plan*.json')):
            index=int(file.stem[4:]);plan=read(file);child=folder/('parent%d'%index)
            for record in read(child/'EXECUTION.json')['records']:
                ident=plan['execution_id'];slot=record['slot'];ix=lookup[(ident,slot,option)]
                assert ix not in rows
                trace=child/plan['parent_id']/record['trace']['file']
                assert sha(trace)==record['trace']['sha256']
                with np.load(trace) as z:q=z['arm_joint_positions'];tip=z['gripper_pose'][:,:3]
                err=np.linalg.norm(forward(q,qref,base,relative)-tip,axis=-1)
                all_errors.append((float(err.max()),float(err.mean()),len(q)))
                rows[ix]=(q[0],)+prefix(record,q,tip)
    assert set(rows)==set(range(len(d['ids'])))
    # Public robot geometry approximation checked against every observed TRAIN
    # step before any learner fit. 5mm is well below the existing2cm floor guard.
    max_error=max(v[0] for v in all_errors)
    if max_error>.005:raise ValueError('Public FK mismatch; retain diagnostic, do not fit:'+str(max_error))
    keys=('initial_q','prefix_q','prefix_tip','prefix_valid','prefix_hazard','prefix_observed')
    for j,k in enumerate(keys):d[k]=np.asarray([rows[i][j] for i in range(len(rows))])
    np.savez_compressed(out/'samples.npz',**d)
    with np.load(model) as z:np.savez_compressed(out/'public_robot.npz',**dict(z))
    write(out/'MANIFEST.json',dict(role='TRAIN',rows=len(rows),source_hashes=hashes,
        dataset_sha256=sha(dataset),samples_sha256=sha(out/'samples.npz'),
        public_robot_sha256=sha(out/'public_robot.npz'),max_FK_tip_error_m=max_error,
        mean_FK_tip_error_m=sum(a*b for _,a,b in all_errors)/sum(b for _,_,b in all_errors),
        completed_prefix_segments=int(d['prefix_valid'].sum()),observed_segments=int(d['prefix_observed'].sum()),
        labels='4actual normalized-time states per completed segment; first finite-budget failure observed, suffix unknown',
        forward_information='current path/current observation/public canonical robot state only; actual q/prefix/hazard TRAIN-only labels',
        no_DEV_feedback=True,locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--expanded',action='store_true');build(**vars(p.parse_args()))
