"""TRAIN-only canonical-branch IK/body probes; failures remain unknown.

These are static configuration probes, not successful robot executions.
No shared environment changes and no deployment geometry input is introduced.
"""
import argparse,copy,os,time
import numpy as np
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha


def callback(task,posts,targets,saved,plan,output,phase,counts,gripper_shapes,external_shapes):
    from scripts import collect_observed_layout_variation as physical
    arm=task._robot.arm;initial=np.asarray(arm.get_joint_positions());quat=np.asarray(saved['initial'].gripper_pose[3:]);records=[];xyz=[];joints=[];status=[]
    paths=np.asarray(plan['probe_drafts']);probes=[]
    for slot,path in enumerate(paths):
        probes.extend((slot,j,p,'draft') for j,p in enumerate(path[1:],1))
        probes.extend((slot,j,path[j]+[0,0,.08],'raised_8cm') for j in (6,8,10,12,14,16,18))
    assert len(probes)==240
    for index,(slot,node,point,kind) in enumerate(probes):
        # Robot state is reset for each probe; zero dynamics/path simulation.
        phase['name']='route';arm.set_joint_positions(initial.tolist(),disable_dynamics=False)
        record=dict(index=index,slot=slot,node=node,kind=kind,requested_xyz=point.tolist(),known=False,branch='Jacobian from registered public canonical joints',executed=False)
        q=np.full(7,np.nan);label=-1;tic=time.monotonic()
        try:
            q=np.asarray(arm.solve_ik_via_jacobian(point.tolist(),quaternion=quat.tolist()));arm.set_joint_positions(q.tolist(),disable_dynamics=False)
            achieved=np.asarray(arm.get_tip().get_position());record['fk_error_m']=float(np.linalg.norm(achieved-point))
            if record['fk_error_m']>.005:raise ValueError('Returned IK pose exceeds5mm verification tolerance')
            collision=physical.legacy.robot_collision_check(task,gripper_shapes,external_shapes)
            record.update(known=True,collision_pair=collision,post_collisions=[p.get_name() for p in posts if arm.check_arm_collision(p)])
            label=int(bool(collision))
        except Exception as error:record['error']=repr(error)
        finally:
            record['seconds']=time.monotonic()-tic;records.append(record);xyz.append(point);joints.append(q);status.append(label)
    phase['name']='route';arm.set_joint_positions(initial.tolist(),disable_dynamics=False)
    phase['name']='audit';np.savez_compressed(output/'body_probes.npz',xyz=xyz,joints=joints,status=status)
    write(output/'BODY_PROBES.json',dict(input_id=plan['probe_id'],role='TRAIN',probes=240,known_safe=status.count(0),known_colliding=status.count(1),unknown=status.count(-1),records=records,public_initial_joints=initial.tolist(),quaternion=quat.tolist(),scope='Static fixed-Jacobian-branch body collision labels; no path execution, no reachability/infeasibility certificate'))
    counts['explicit_ik_queries']=240
    return [],[],False


def collect(name,data_name='interventions_calibrated_targets_v1',family_start=0,families=2):
    from scripts import collect_observed_layout_variation as physical
    from scripts.collect_observed_layout_hash_recovery import validate_initial_geometry
    data=ROOT/'data/selective_repair_interventions_v1';registration=read(data/'registration.json')
    chosen=sorted({p['family_id'] for p in registration['parent_plan'] if p['role']=='TRAIN'})[family_start:family_start+families]
    plans=[p for p in registration['parent_plan'] if p['family_id'] in chosen and p['variant'] in ('open','closed')]
    assert len(plans)==2*families and all(p['role']=='TRAIN' for p in plans)
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    with np.load(RUN/data_name/'samples.npz') as z:
        ix=z['splits']=='TRAIN';ids=z['ids'][ix];drafts=z['drafts'][ix]
    byid={str(v):i for i,v in enumerate(ids)}
    manifest=dict(source_commit=os.environ.get('CODE_COMMIT'),source_sha256={str(f.relative_to(SOURCE)):sha(f) for d in ('scripts','configs','research_selective_repair_v1') for f in (SOURCE/d).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh')},samples_sha256=sha(RUN/data_name/'samples.npz'),TRAIN_families=chosen,variants=['open','closed'],probes_per_parent=240,init_ik_queries=0,path_queries=0,simulation_after_probe=0,teacher_scope='Static canonical branch, unknown failures excluded from collision classifier',locked_access=False)
    write(out/'MANIFEST.json',manifest);physical.collect_routes=callback;physical.validate_initial_geometry=validate_initial_geometry;physical.PROTOCOL='selective_body_probe_train_v1'
    summaries=[]
    for i,original in enumerate(plans):
        plan=copy.deepcopy(original);ident=plan['parent_id']+'_target0';assert ident in byid
        plan.update(probe_id=ident,probe_drafts=drafts[byid[ident]].tolist());file=out/('plan%d.json'%i);write(file,plan)
        dest=out/('parent%d'%i);receipt=physical.physical_worker(registration,manifest,plan,file,dest)
        if receipt['status']=='error':raise RuntimeError('Technical probe failure retained: '+str(receipt['fatal_error']))
        summary=read(dest/'BODY_PROBES.json');summaries.append({k:v for k,v in summary.items() if k!='records'});print(summaries[-1],flush=True)
    write(out/'SUMMARY.json',dict(parents=len(plans),probes=240*len(plans),known_safe=sum(s['known_safe'] for s in summaries),known_colliding=sum(s['known_colliding'] for s in summaries),unknown=sum(s['unknown'] for s in summaries),parents_summary=summaries,static_probes=True,actual_executions=0,locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--data-name',default='interventions_calibrated_targets_v1');p.add_argument('--family-start',type=int,default=0);p.add_argument('--families',type=int,default=2);collect(**vars(p.parse_args()))
