"""TRAIN fixed-planner-budget outcomes; failure is not infeasibility.

This replaces uninformative distant single-Jacobian pose labels. A found path
is a planning outcome only: no simulation or full execution success is claimed.
"""
import argparse,random,time
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,read,write
from research_selective_repair_v1 import body_teacher


def callback(task,posts,targets,saved,plan,output,phase,counts,gripper_shapes,external_shapes):
    arm=task._robot.arm;initial=np.asarray(arm.get_joint_positions());quat=np.asarray(saved['initial'].gripper_pose[3:]);records=[];probes=[]
    paths=np.asarray(plan['probe_drafts'])
    for slot,path in enumerate(paths):
        probes.extend((slot,j,path[j],'draft') for j in (2,4,6,8,10,12,14,16,18,22))
        probes.extend((slot,j,path[j]+[0,0,.08],'raised_8cm') for j in (6,8,10,12,14))
    assert len(probes)==120
    for index,(slot,node,point,kind) in enumerate(probes):
        phase['name']='route';arm.set_joint_positions(initial.tolist(),disable_dynamics=False)
        seed=plan['seed']*1000+index;random.seed(seed);np.random.seed(seed)
        record=dict(index=index,slot=slot,node=node,kind=kind,requested_xyz=point.tolist(),planner_seed=seed,found_path=False,executed=False,retry_budget=0,get_path_calls=1,ignore_collisions=False)
        counts['get_path_calls']=counts.get('get_path_calls',0)+1;tic=time.monotonic()
        try:
            path=arm.get_path(point.tolist(),quaternion=quat.tolist(),ignore_collisions=False)
            record.update(found_path=True,path_class=type(path).__name__)
        except Exception as error:record['finite_planning_error']=repr(error)
        finally:record['seconds']=time.monotonic()-tic;records.append(record)
    phase['name']='route';arm.set_joint_positions(initial.tolist(),disable_dynamics=False)
    phase['name']='audit';np.savez_compressed(output/'planning_probes.npz',xyz=[p[2] for p in probes],found_path=[r['found_path'] for r in records])
    found=sum(r['found_path'] for r in records)
    report=dict(input_id=plan['probe_id'],role='TRAIN',probes=120,known_safe=0,known_colliding=0,unknown=120,found_path=found,finite_planning_failure=120-found,records=records,public_initial_joints=initial.tolist(),quaternion=quat.tolist(),scope='Fixed-budget collision-aware get_path outcomes. Found path is not an executed trajectory; failure is not IK or geometric infeasibility. No dynamics.')
    write(output/'BODY_PROBES.json',report);write(output/'PLANNING_PROBES.json',report)
    return [],[],False


def collect(name,family_start=0,families=2):
    body_teacher.callback=callback
    body_teacher.collect(name,family_start=family_start,families=families,teacher_kind='planner')
    out=RUN/name;manifest=read(out/'MANIFEST.json');manifest.update(teacher_scope='Fixed budget get_path outcome likelihood; neither reachability nor executed success label',probes_per_parent=120,path_queries_per_parent=120,explicit_ik_queries_not_separately_invoked=True);write(out/'PLANNER_MANIFEST.json',manifest)
    reports=[read(p) for p in sorted(out.glob('parent*/PLANNING_PROBES.json'))]
    paired=[]
    for report in reports:
        base={(r['slot'],r['node']):r for r in report['records'] if r['kind']=='draft'}
        paired.extend((base[r['slot'],r['node']]['found_path'],r['found_path']) for r in report['records'] if r['kind']=='raised_8cm')
    summary=dict(parents=len(reports),probes=120*len(reports),found_path=sum(r['found_path'] for r in reports),finite_planning_failure=sum(r['finite_planning_failure'] for r in reports),raised_pairs=len(paired),raised_recovers=sum(not a and b for a,b in paired),raised_loses=sum(a and not b for a,b in paired),actual_executions=0,scope=manifest['teacher_scope'],locked_access=False)
    write(out/'PLANNING_SUMMARY.json',summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--family-start',type=int,default=0);p.add_argument('--families',type=int,default=2);collect(**vars(p.parse_args()))
