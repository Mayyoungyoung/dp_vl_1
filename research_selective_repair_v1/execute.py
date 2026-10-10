"""Fixed Panda executor on actual returned candidates, including every failure."""
import argparse,copy,os,random,time,traceback
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,DATA,read,write,sha,lines

def callback(task,posts,targets,saved,plan,output,phase,counts,gripper_shapes,external_shapes):
    from scripts import collect_observed_layout_variation as physical
    pilot=physical.pilot;legacy=physical.legacy;parent=plan['parent_id'];folder=output/parent
    folder.mkdir(exist_ok=True);cfg=plan['config'];selected=plan['execution_selected'];paths=np.asarray(plan['execution_paths'])
    records=[];witnesses=[];fatal=False
    for rank,slot in enumerate(selected):
        record=dict(input_id=plan['execution_id'],slot=int(slot),rank=rank,attempted=True,success=False,simulated_steps=0,planning_segments=[],executor='unchanged pilot.execute/get_path(ignore_collisions=False)',retry_budget=0)
        trace=[];counts['route_attempts']+=1;physical.durable_slot(output/'slot_ledger.jsonl','started',parent,plan['execution_id'],rank)
        tic=time.monotonic()
        try:
            phase['name']='restore';check,_,obs=physical.physical.restore_check(task,posts,saved,gripper_shapes,external_shapes);record['restore']=check;counts['strict_route_restores']+=1
            trace.append(pilot.state_sample(task))
            if not np.allclose(trace[0][0][:3],paths[slot,0],atol=.005,rtol=0):
                raise ValueError('Route start does not match restored public gripper state')
            # Equal RNG per request/rank for every method; no method-specific retry.
            random.seed(plan['seed']+rank);np.random.seed(plan['seed']+rank)
            phase['name']='route';pilot.execute(task,paths[slot,1:],trace[0][0][3:],gripper_shapes,external_shapes,trace,record,cfg)
            phase['name']='audit';actual=task._robot.arm.get_tip().get_pose();goal=np.asarray(cfg['goal_xyz'])[plan['execution_target']]
            record['goal_error_m']=float(np.linalg.norm(np.asarray(actual)[:3]-goal));record['success']=record['goal_error_m']<=.03
            record['failure_type']=None if record['success'] else 'goal_tracking'
        except Exception as error:
            record['error']=repr(error);record['traceback']=traceback.format_exc()
            record['failure_type']='collision' if record.get('collision_pair') else 'planning_IK_or_path' if record['planning_segments'] and record['planning_segments'][-1].get('planning_status')=='failed' else 'tracking_or_restore'
            if phase['name']=='restore':fatal=True
        finally:
            phase['name']='audit';record['seconds']=time.monotonic()-tic
            record['trace']=pilot.save_trace(folder/('execution_rank%d.npz'%rank),trace)
            if len(trace)>1:
                joints=np.asarray([x[2] for x in trace]);record['maximum_joint_step_rad']=float(np.max(np.abs(np.diff(joints,axis=0))))
                record['max_tip_waypoint_tracking_error_m']=None # segments retain achieved trace; no fabricated nearest-index match
            pilot.add_execution_totals(counts,record);legacy.append_json(output/'execution.jsonl',record)
            physical.durable_slot(output/'slot_ledger.jsonl','completed',parent,plan['execution_id'],rank)
            if record['success']:counts['accepted_routes']+=1
            records.append(record)
        if fatal:break
    write(output/'EXECUTION.json',dict(request=plan['execution_id'],actual_returned_indices=selected,records=records,requested=len(selected),attempted=len(records),success=sum(x['success'] for x in records),full_arm_simulation=True,held_object=False,scope='Reach gripper task; actual collision-aware planner/controller and explicit arm/gripper collision monitoring. No held-object or continuous collision certificate.'))
    return witnesses,[dict(id=plan['execution_id'],accepted=sum(r['success'] for r in records),requested=len(selected))],fatal

def run(name,pool,rows,data=None,limit=4,all_candidates=False):
    from scripts import paired_modes_data as paired
    from scripts import collect_observed_layout_variation as physical
    from scripts.collect_observed_layout_hash_recovery import validate_initial_geometry
    data=data or DATA;registration=read(data/'registration.json');plans={p['parent_id']:p for p in registration['parent_plan']}
    with np.load(pool) as z:arrays={k:z[k] for k in ('ids','paths')}
    metrics=read(rows);byid={r['id']:r for r in metrics};index={str(x):i for i,x in enumerate(arrays['ids'])}
    # Prospectively fixed: first two DEV families, open/closed target0; all actual returned4.
    families=sorted({p['family_id'] for p in plans.values() if p['role']=='DEV_MODEL'})[:2]
    ids=[p['parent_id']+'_target0' for p in plans.values() if p['family_id'] in families and p['variant'] in ('open','closed')][:limit]
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);source={str(f.relative_to(SOURCE)):sha(f) for d in ('scripts','configs','research_selective_repair_v1') for f in (SOURCE/d).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh')}
    physical.collect_routes=callback;physical.validate_initial_geometry=validate_initial_geometry;physical.PROTOCOL='selective_repair_fixed_executor_v1'
    results=[]
    for j,ident in enumerate(ids):
        parent=ident.rsplit('_target',1)[0];plan=copy.deepcopy(plans[parent]);plan.update(execution_id=ident,execution_target=0,execution_paths=arrays['paths'][index[ident]].tolist(),execution_selected=list(range(8)) if all_candidates else byid[ident]['selected_indices'])
        cfg=out/('plan%d.json'%j);write(cfg,plan);dest=out/('request%d'%j)
        r=physical.physical_worker(registration,dict(source_sha256=source),plan,cfg,dest)
        if r['status']=='error':raise RuntimeError('Technical execution failure, preserve and recover: '+str(r['fatal_error']))
        results.append(read(dest/'EXECUTION.json'))
    write(out/'RESULTS.json',dict(requests=len(ids),requested_routes=(8 if all_candidates else 4)*len(ids),success=sum(r['success'] for r in results),records=results,pool_sha256=sha(pool),rows_sha256=sha(rows),source_commit=os.environ.get('CODE_COMMIT'),locked_access=False,all_candidate_damage_cohort=all_candidates,subset='First two DEV families,open/closed,target0; fixed8slot common body-damage cohort,not returned4' if all_candidates else 'first two DEV families, open/closed target0, actual returned4; frozen before looking at execution outcomes'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--pool',required=True,type=Path);p.add_argument('--rows',required=True,type=Path);p.add_argument('--data',type=Path);p.add_argument('--limit',type=int,default=4);p.add_argument('--all-candidates',action='store_true');run(**vars(p.parse_args()))
