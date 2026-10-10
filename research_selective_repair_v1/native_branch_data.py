"""Explicit TRAIN native-joint label contract; no approximate FK dependency.

This is a NEW supervised label contract, not permission to reinterpret an old
event-only fit. Native completed endpoint joints are labels, never forward input.
The previous approximate-FK 5mm state-model gate remains unchanged.
"""
import argparse
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_prefix_data import prefix,crossing_events
from research_selective_repair_v1.body_feedback_data import WORDS

def build(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    base=RUN/'body_events_data_v1';manifest=read(base/'MANIFEST.json')
    assert manifest['rows']==2304 and manifest['event_only'] and manifest['no_DEV_feedback']
    with np.load(base/'samples.npz') as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    lookup={str(v):i for i,v in enumerate(d['ids'])};rows=[];hashes={}
    sources=[('body_prefix_null_TRAIN_teacher_probe_v1',4),('body_prefix_loop_TRAIN_teacher_probe_v1',5),
             ('body_prefix_native_TRAIN_fit_support_v1_null',4),('body_prefix_native_TRAIN_fit_support_v1_loop',5)]
    for source,option in sources:
        folder=RUN/source;m=read(folder/'MANIFEST.json');assert m['TRAIN_feedback_only'] and m['locked_access'] is False
        assert (folder/'SUMMARY.json').exists()
        hashes[source+'/MANIFEST.json']=sha(folder/'MANIFEST.json')
        labels={(r['id'],r['slot']):r for r in read(RUN/(source+'_semantics_v1')/'ROWS.json')}
        for file in sorted(folder.glob('plan*.json')):
            j=int(file.stem[4:]);plan=read(file);ident=plan['execution_id'];child=folder/('parent%d'%j)
            paths=np.asarray(plan['execution_paths'],np.float32)
            for record in read(child/'EXECUTION.json')['records']:
                slot=record['slot'];trace=child/plan['parent_id']/record['trace']['file'];assert sha(trace)==record['trace']['sha256']
                row={k:v[lookup[ident]].copy() for k,v in d.items()}
                w=labels[(ident,slot)]['successful_executable_word']
                row.update(ids=np.asarray(ident),slots=np.asarray(slot),options=np.asarray(option),paths=paths[slot],labels=np.asarray(0 if w is None else WORDS.index(w)+1))
                with np.load(trace) as z:q=z['arm_joint_positions'];tip=z['gripper_pose'][:,:3]
                row.update(zip(('initial_q','prefix_q','prefix_tip','prefix_valid','prefix_hazard','prefix_observed','event_q','event_tip','event_present','event_observed'),
                               (q[0],)+prefix(record,q,tip)+crossing_events(record,q,tip,plan['config']['row_x'])))
                rows.append(row)
    assert len(rows)==192
    combined={k:np.concatenate([v,np.asarray([r[k] for r in rows],dtype=v.dtype)]) for k,v in d.items()}
    contract=RUN/'public_panda_joint_contract_v2/PUBLIC_JOINT_CONTRACT.json';robot=read(contract)
    lo=np.asarray(robot['lower']);hi=np.asarray(robot['upper']);joint=combined['prefix_q'][:,:,-1]
    within=((joint>=lo-1e-5)&(joint<=hi+1e-5)).all(-1)
    combined['joint_label_valid']=combined['prefix_valid']&within
    assert np.max(np.abs(combined['initial_q']-robot['q0']))<1e-5,'This first implementation uses the registered common public initial state'
    np.savez_compressed(out/'samples.npz',**combined)
    write(out/'MANIFEST.json',dict(rows=len(combined['ids']),samples_sha256=sha(out/'samples.npz'),base_dataset_sha256=sha(base/'samples.npz'),
        teacher_manifest_sha256=hashes,public_joint_contract=robot,public_joint_contract_sha256=sha(contract),
        native_joint_labels=True,no_DEV_feedback=True,locked_access=False,
        completed_joint_labels=int(combined['joint_label_valid'].sum()),excluded_outside_public_intervals=int((combined['prefix_valid']&~within).sum()),
        label_contract='Native completed endpoint joints, observed hazards and measured crossing positions only. No FK/IK targets, no actual future state in forward. Unknown and outside-interval endpoints masked.',
        previous_FK_gate='Unchanged 5mm gate for previous approximate-FK model; this FK-free model does not use that representation.',
        support='2304 original TRAIN routes +192 null/loop target0 observations, fit families0..3 and diagnostic14..15; partial loop support, not full allgoal data',
        native_randomness_controlled=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);build(**vars(p.parse_args()))
