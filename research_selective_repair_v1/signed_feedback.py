"""Targeted signed preimage support, same labels for every competing learner."""
import argparse
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,read,write,sha
from research_selective_repair_v1.body_options import options
from research_selective_repair_v1.body_prefix_data import prefix,crossing_events
from research_selective_repair_v1.body_feedback_data import WORDS

SOURCE='body_execution_train_signed_lower_pilot_v1'
DATASET='body_signed_events_pilot_v1'

def lowering_amplitude():
    artifact=RUN/'body_crossing_constant_TRAIN_v1/SUMMARY.json'
    d=read(artifact)
    assert d['locked_access'] is False and 'TRAIN' in d['scope']
    return -float(d['bias_yz_m'][0][1])

def build(name=DATASET):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    base=RUN/'body_events_data_v1';manifest=read(base/'MANIFEST.json')
    assert manifest['rows']==2304 and manifest['event_only'] and manifest['no_DEV_feedback']
    with np.load(base/'samples.npz') as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    byid={str(v):i for i,v in enumerate(d['ids'])}
    pool=RUN/'constraints_global_interventions_TRAIN_body_v1'
    with np.load(pool/'sealed_predictions.npz') as z:pred={k:z[k] for k in ('ids','paths','completed')}
    ix={str(v):i for i,v in enumerate(pred['ids'])}
    teacher=RUN/SOURCE;tm=read(teacher/'MANIFEST.json');audit=RUN/(SOURCE+'_semantics_v1')
    assert tm['TRAIN_feedback_only'] and tm['pool_sha256']==sha(pool/'pool.npz')
    assert tm['target_ids']==[0] and np.isclose(tm['lift_m'],lowering_amplitude(),atol=1e-12)
    assert (teacher/'SUMMARY.json').exists()
    labels={(r['id'],r['slot']):r for r in read(audit/'ROWS.json')};rows=[];trace_hashes=[]
    for file in sorted(teacher.glob('plan*.json')):
        j=int(file.stem[4:]);plan=read(file);ident=plan['execution_id'];child=teacher/('parent%d'%j)
        expected=options(pred['paths'][ix[ident]],pred['completed'][ix[ident]],lowering_amplitude())[:,1]
        paths=np.asarray(plan['execution_paths'],np.float32);np.testing.assert_allclose(paths,expected,atol=1e-7,rtol=0)
        for r in read(child/'EXECUTION.json')['records']:
            slot=r['slot'];trace=child/plan['parent_id']/r['trace']['file'];assert sha(trace)==r['trace']['sha256']
            label=labels[(ident,slot)];w=label['successful_executable_word'];target=0 if w is None else WORDS.index(w)+1
            row={k:v[byid[ident]].copy() for k,v in d.items()}
            row.update(ids=np.asarray(ident),slots=np.asarray(slot),options=np.asarray(3),paths=paths[slot],labels=np.asarray(target))
            with np.load(trace) as z:q=z['arm_joint_positions'];tip=z['gripper_pose'][:,:3]
            keys=('initial_q','prefix_q','prefix_tip','prefix_valid','prefix_hazard','prefix_observed','event_q','event_tip','event_present','event_observed')
            values=(q[0],)+prefix(r,q,tip)+crossing_events(r,q,tip,plan['config']['row_x'])
            row.update(zip(keys,values));rows.append(row);trace_hashes.append(sha(trace))
    assert len(rows)==96 and len(set(str(r['families']) for r in rows))==6
    combined={k:np.concatenate([v,np.asarray([r[k] for r in rows],dtype=v.dtype)]) for k,v in d.items()}
    assert len(combined['ids'])==2400 and set(combined['roles'])=={'TRAIN'}
    np.savez_compressed(out/'samples.npz',**combined)
    write(out/'MANIFEST.json',dict(rows=2400,samples_sha256=sha(out/'samples.npz'),
        base_dataset_sha256=sha(base/'samples.npz'),base_manifest_sha256=sha(base/'MANIFEST.json'),
        additional_teacher_manifest_sha256=sha(teacher/'MANIFEST.json'),additional_audit_sha256=sha(audit/'ROWS.json'),trace_sha256=trace_hashes,
        event_only=True,no_DEV_feedback=True,locked_access=False,lowering_amplitude_m=lowering_amplitude(),
        event_only_contract='Measured crossing/tip/hazard labels only; q/FK metadata is not learner input/supervision. Original5mmFK state gate unchanged.',
        new_support='96target0 lowering observations,TRAINfamilies0..3/14..15; partial signed-support pilot,not complete allgoal lowering feedback',
        diagnostic_TRAIN_families=['sr_family_642014','sr_family_642015'],deployment_options=['identity','uniform8cm_lift','TRAIN_mean_crossing_error_lowering'],
        forward_information='Current draft/current observation/public initial robot state only; no actual event/native query',
        matching='Every learner receives identical2400rows; previous positive preserved recipe stays shared training feedback but is replaced by lower in three-option pilot inference'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default=DATASET);build(**vars(p.parse_args()))
