"""Counterbalanced TRAIN native test of one bounded prefix loop vs null.

This is a mechanism falsification, not a learned deployment or causal theorem.
Native sampling randomness is not seeded by the Python/NumPy seed contract.
"""
import argparse,copy,os,time
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,SOURCE,read,write,sha
from research_selective_repair_v1.prefix_conditioning import proposals
from research_selective_repair_v1.execute import callback
from research_selective_repair_v1.execution_semantics import audit

def run(name):
    from scripts import collect_observed_layout_variation as physical
    from scripts.collect_observed_layout_hash_recovery import validate_initial_geometry
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    registration=read(ROOT/'data/selective_repair_interventions_v1/registration.json')
    families=sorted({p['family_id'] for p in registration['parent_plan'] if p['role']=='TRAIN'})[-2:]
    plans=sorted([p for p in registration['parent_plan'] if p['family_id'] in families and p['variant'] in ('open','closed')],key=lambda p:p['parent_id'])
    assert len(plans)==4
    source=RUN/'constraints_global_interventions_TRAIN_body_v1';seal=read(source/'SEAL.json')
    assert seal['role']=='TRAIN' and seal['current_observation_only'] and sha(source/'sealed_predictions.npz')==seal['prediction_sha256']
    with np.load(source/'sealed_predictions.npz') as z:data={k:z[k] for k in ('ids','paths','completed')}
    index={str(v):i for i,v in enumerate(data['ids'])};base=read(RUN/'constraints_seed0_v1/config.json')['post_base']
    sealed=[]
    for plan in plans:
        ident=plan['parent_id']+'_target0';i=index[ident]
        candidates,eligible,reasons=proposals(data['paths'][i],data['completed'][i],base)
        sealed.append(dict(id=ident,candidates=candidates,eligible=eligible,reasons=reasons))
    np.savez_compressed(out/'SEALED_NN_PROPOSALS.npz',ids=[r['id'] for r in sealed],paths=[r['candidates'] for r in sealed],eligible=[r['eligible'] for r in sealed])
    source_hashes={str(p.relative_to(SOURCE)):sha(p) for directory in ('scripts','research_selective_repair_v1') for p in (SOURCE/directory).rglob('*') if p.suffix in ('.py','.sh')}
    schedule=[('null','loop'),('loop','null'),('loop','null'),('null','loop')]
    write(out/'PROTOCOL.json',dict(source_commit=os.environ.get('CODE_COMMIT'),families=families,requests=4,
        schedule=schedule,attempts_per_arm=32,proposal_sha256=sha(out/'SEALED_NN_PROPOSALS.npz'),
        current_observation_only_proposals=True,eligible_slots=int(sum(r['eligible'].sum() for r in sealed)),
        task='TwoTRAINheld families,open/closed,target0,all8; no DEV/returned4/new-layout claim',
        controller='Unchanged pilot.execute/SBL/native budgets,23segments,retries0',native_RNG_controlled=False,locked_access=False))
    folders={};records={label:[] for label in ('null','loop')}
    for label in records:
        folder=RUN/('body_prefix_%s_TRAIN_teacher_probe_v1'%label);folder.mkdir(parents=True,exist_ok=False);folders[label]=folder
        write(folder/'MANIFEST.json',dict(source_commit=os.environ.get('CODE_COMMIT'),source_sha256=source_hashes,
            pool_sha256=sha(source/'pool.npz'),proposal_sha256=sha(out/'SEALED_NN_PROPOSALS.npz'),
            TRAIN_feedback_only=True,locked_access=False,recipe=label,teacher_routes_per_parent=8))
    physical.collect_routes=callback;physical.validate_initial_geometry=validate_initial_geometry;physical.PROTOCOL='bounded_prefix_native_probe_v1'
    for j,(original,pre) in enumerate(zip(plans,sealed)):
        for label in schedule[j]:
            folder=folders[label];option=1 if label=='null' else 2
            plan=copy.deepcopy(original);plan.update(execution_id=pre['id'],execution_target=0,execution_paths=pre['candidates'][:,option].tolist(),execution_selected=list(range(8)))
            file=folder/('plan%d.json'%j);write(file,plan);dest=folder/('parent%d'%j)
            result=physical.physical_worker(registration,dict(source_sha256=source_hashes),plan,file,dest)
            assert result['status']=='collection_finished',result
            records[label].append(read(dest/'EXECUTION.json'))
            print(dict(id=pre['id'],recipe=label,success=records[label][-1]['success'],eligible=int(pre['eligible'].sum())),flush=True)
    summaries=[];prefix_rows=[]
    for label,folder in folders.items():
        write(folder/'SUMMARY.json',dict(attempted=32,success=sum(r['success'] for r in records[label]),records=records[label],TRAIN_feedback_only=True,locked_access=False))
        audit(folder.name+'_semantics_v1',folder.name);s=read(RUN/(folder.name+'_semantics_v1')/'SUMMARY.json')
        summaries.append(dict(recipe=label,E8=s['mean_executable_distinct'],successful_clear=s['successful_clear_routes'],attempted=32,requests=s['requests_results']))
        for j,(original,pre) in enumerate(zip(plans,sealed)):
            child=folder/('parent%d'%j)
            for record in read(child/'EXECUTION.json')['records']:
                slot=record['slot'];segments=record['planning_segments'];complete=len(segments)>=3 and all(seg['simulation_status']=='success' for seg in segments[:3])
                row=dict(id=pre['id'],slot=slot,recipe=label,eligible=bool(pre['eligible'][slot]),first_three_complete=complete,full_success=record['success'])
                if complete:
                    trace=child/original['parent_id']/record['trace']['file'];assert sha(trace)==record['trace']['sha256']
                    with np.load(trace) as z:q=z['arm_joint_positions'];tip=z['gripper_pose'][:,:3]
                    end=sum(seg['simulated_steps'] for seg in segments[:3])
                    row.update(returned_root_error_m=float(np.linalg.norm(tip[end]-tip[0])),joint_change_rad=float(np.linalg.norm(q[end]-q[0])),initial_q=q[0].tolist(),after_prefix_q=q[end].tolist())
                prefix_rows.append(row)
    write(out/'SUMMARY.json',dict(records=summaries,prefix_rows=prefix_rows,seconds=time.monotonic()-tic,
        scope='Counterbalanced TRAIN all8 mechanism probe; source proposals sealed before native/truth; public root/state future labels only. Native timing/randomness and two-family sample do not establish causal generalization',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
