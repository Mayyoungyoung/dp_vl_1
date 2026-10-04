"""New paired RGB-D observations with explicitly geometric upper-level teachers.

Existing raw data and collectors remain unchanged. Each scene is rendered in
RLBench from the existing canonical robot state; no robot rollout is claimed.
All variants of a family have one role and share goals, colors and robot state.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from scripts import observed_layout_variation as geom
from scripts import collect_observed_layout_variation as physical
from scripts.collect_observed_layout_hash_recovery import quantized_hash_v2, validate_initial_geometry
from scripts.run_observed_probability import ROOT, SOURCE, read, write, sha, lines

DATA = ROOT/'data/paired_modes_v1'
RUN = ROOT/'runs/paired_modes_v1'
POLICY = SOURCE/'configs/paired_modes_v1.json'


def registration():
    policy = read(POLICY)
    rng = np.random.RandomState(policy['seed'])
    old = read(SOURCE/'configs/observed_two_row_extension288_v1.json')
    colors = geom.prior.color_table()
    plans = []
    for family in range(policy['train_families'] + policy['dev_families']):
        role = 'TRAIN' if family < policy['train_families'] else 'DEV_MODEL'
        family_id = 'paired_family_%06d' % (520000+family)
        row_x = [float(rng.uniform(.15,.21)), float(rng.uniform(.32,.37))]
        offsets = rng.uniform(-.025,.025,2)
        widths = rng.uniform(.085,.125,2)
        ys = [[float(o-w),float(o+w)] for o,w in zip(offsets,widths)]
        heights = [float(rng.uniform(.12,.16)),float(rng.uniform(.12,.16))]
        goals = [[float(rng.uniform(.45,.49)),float(rng.uniform(y-.015,y+.015)),.84]
                 for y in (-.16,0.,.16)]
        palette = [copy.deepcopy(colors[int(i)]) for i in rng.choice(len(colors),3,replace=False)]
        changed_row = family % 2
        for variant_i, variant in enumerate(policy['variants']):
            c = dict(protocol='paired_modes_v1',split=role,seed=520000+3*family+variant_i,
                     row_x=copy.deepcopy(row_x),post_y=copy.deepcopy(ys),post_heights=heights,
                     post_base_z=.755,entry_xyz=copy.deepcopy(old['canonical_init']['canonical_entry_xyz']),
                     goal_xyz=goals,tip_clearance_m=.02,endpoint_tolerance_m=.03,
                     maximum_steps_per_segment=1000,requested_setup_actions=0,
                     requested_route_proposals=27,preparation_xyz=[])
            if variant == 'closed':
                # Physical posts do not overlap, but their .02m inflated bands do.
                c['post_y'][changed_row] = [float(offsets[changed_row]-.028),float(offsets[changed_row]+.028)]
            elif variant == 'shifted':
                c['post_y'][changed_row] = [y+.018 for y in c['post_y'][changed_row]]
            cs,hs = geom.geometry(c)
            plans.append(dict(index=len(plans),parent_id=family_id+'_'+variant,
                family_id=family_id,variant=variant,changed_row=changed_row,
                role=role,split=role,seed=c['seed'],config=c,target_colors=palette,
                registered_geometry_1mm_sha256=quantized_hash_v2(cs,hs,goals),
                guide_plans=[geom.guide_plan(c,t) for t in range(3)],
                geometry_precheck=geom.geometry_precheck(c),low_gap_certificates=geom.gap_certificates(c)))
    hashes = [p['registered_geometry_1mm_sha256'] for p in plans]
    assert len(set(hashes)) == len(hashes)
    assert all(p['geometry_precheck']['passed'] for p in plans)
    return dict(protocol='paired_modes_v1',policy=policy,parent_plan=plans,
                canonical_init=old['canonical_init'],specification=old,
                source_commit=os.environ.get('CODE_COMMIT'),
                reference_kind=policy['reference_kind'])


def prepare():
    DATA.mkdir(parents=True,exist_ok=False)
    write(DATA/'registration.json',registration())
    write(DATA/'source_sha256.json',{str(p.relative_to(SOURCE)):sha(p)
          for directory in ('scripts','routeset','configs') for p in (SOURCE/directory).rglob('*')
          if p.is_file() and p.suffix in ('.py','.json','.sh')})


def geometric_references(task,posts,targets,saved,plan,output,phase,counts,*unused):
    """No dynamics calls: certify proposed full polylines and resampled H24."""
    from PIL import Image
    c=plan['config']; parent=plan['parent_id']; folder=output/parent
    obs=saved['initial']; cs,hs=geom.geometry(c); goals=np.asarray([t.get_position() for t in targets])
    Image.fromarray(obs.front_rgb).save(folder/'front.png')
    np.savez_compressed(folder/'observation.npz',depth=obs.front_depth,gripper_pose=obs.gripper_pose,
        gripper_open=obs.gripper_open,camera_intrinsics=obs.misc['front_camera_intrinsics'],
        camera_extrinsics=obs.misc['front_camera_extrinsics'])
    np.savez_compressed(folder/'verification_only.npz',obstacle_centers=cs,obstacle_halfsizes=hs,
                       target_centers=goals,mask=obs.front_mask)
    witnesses=[]; summaries=[]
    for target in range(3):
        identifier=parent+'_target%d'%target; routes=[]; types=[]
        physical.legacy.append_json(output/'observations.jsonl',dict(id=identifier,parent_id=parent,
            split=plan['role'],image=parent+'/front.png',instruction=
            'Move the gripper to touch the %s sphere while avoiding the gray posts.'%plan['target_colors'][target]['name']))
        for slot in plan['guide_plans'][target]:
            counts['route_attempts']+=1
            attempt=slot['slot']; physical.durable_slot(output/'slot_ledger.jsonl','started',parent,identifier,attempt)
            rec=dict(input_id=identifier,attempt=attempt,success=False,reference_kind='geometric_teacher',
                     proposed_type=slot['intent_supervision_only'],robot_executed=False,simulated_steps=0)
            if not slot['collection_allowed']:
                rec['reason']=slot['precheck_reason']
            else:
                xyz=np.vstack([obs.gripper_pose[:3],slot['waypoints_supervision_only']])
                fields,h24,passed=physical.route_acceptance(xyz,goals,target,c)
                rec.update(fields); rec['success']=bool(passed)
                if passed:
                    name='target%d_route%d.npz'%(target,attempt)
                    poses=np.column_stack([xyz,np.tile(obs.gripper_pose[3:],(len(xyz),1))])
                    np.savez_compressed(folder/name,gripper_pose=poses,
                        gripper_open=np.full(len(xyz),obs.gripper_open.item()),xyz_24=h24)
                    routes.append(parent+'/'+name); types.append(fields['actual_route_type'])
                    witnesses.append(dict(input_id=identifier,slot=attempt,actual_route_type=fields['actual_route_type'],
                                          file=parent+'/'+name,sha256=sha(folder/name)))
                    counts['accepted_routes']+=1
                else: rec['reason']='full_polyline_or_H24_invalid'
            physical.legacy.append_json(output/'attempts.jsonl',rec)
            physical.durable_slot(output/'slot_ledger.jsonl','completed',parent,identifier,attempt)
        physical.legacy.append_json(output/'supervision.jsonl',dict(id=identifier,parent_id=parent,
            split=plan['role'],task='rlbench_derived_paired_geometric_reach',observation=parent+'/observation.npz',
            routes=routes,route_types=types,verification_only=parent+'/verification_only.npz',
            route_config='route_configs/'+parent+'.json',reference_set_complete=False,
            reference_kind='geometric_teacher_not_robot_rollout',
            semantic_targets=dict(centers=goals.tolist(),target_index=target,tolerance=.03)))
        summaries.append(dict(id=identifier,accepted=len(routes),requested=9))
    return witnesses,summaries,False


def worker(index):
    value=read(DATA/'registration.json'); plan=value['parent_plan'][index]
    expected=read(DATA/'source_sha256.json')
    for name,digest in expected.items():
        if sha(SOURCE/name)!=digest: raise ValueError('Collector immutable source mismatch: '+name)
    # Scoped replacement of callbacks in this new worker process only.
    physical.collect_routes=geometric_references
    physical.validate_initial_geometry=validate_initial_geometry
    physical.PROTOCOL='paired_modes_geometric_teacher_v1'
    cfg=DATA/'parent_configs'/('%03d.json'%index)
    write(cfg,plan)
    out=DATA/'parents'/plan['role']/plan['parent_id']
    result=physical.physical_worker(value,dict(source_sha256=expected),plan,cfg,out)
    # The reused rendering worker has historical TRAIN metadata literals.
    # Finalize this NEW output's metadata to the prospectively assigned role.
    for name in ('manifest.json','mechanical_receipt.json'):
        if (out/name).exists():
            record=read(out/name);record.update(role=plan['role'],reference_kind='geometric_teacher_not_robot_execution')
            write(out/name,record)
    write(out/'reference_contract.json',dict(reference_kind='geometric_teacher_not_robot_execution',
         role=plan['role'],family_id=plan['family_id'],canonical_robot_state=value['canonical_init']['canonical_arm_joints']))
    write(out/'artifact_hashes.json',{p.relative_to(out).as_posix():sha(p)
          for p in out.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    return result


def collect(start,stop,resume=False):
    value=read(DATA/'registration.json'); plans=value['parent_plan'][start:stop]
    work=RUN/('collection_%03d_%03d'%(start,stop))
    work.mkdir(parents=True,exist_ok=resume)
    def one(plan):
        i=plan['index']; receipt=work/('%03d.json'%i)
        if receipt.exists():
            r=read(receipt)
            if r.get('status') in ('completed','failed'):return r
            raise RuntimeError('Unclosed worker; inspect before resume')
        output=DATA/'parents'/plan['role']/plan['parent_id']
        if output.exists():raise FileExistsError('Never replay a started scene: '+str(output))
        if __import__('shutil').disk_usage(str(ROOT)).free < 20*1024**3:raise RuntimeError('Low free space')
        cmd=[str(ROOT/'.venv-sim/bin/python'),'-m','scripts.paired_modes_data','worker','--index',str(i)]
        tic=time.monotonic(); r=dict(index=i,command=cmd,status='running',start_unix=time.time())
        with (work/('%03d.log'%i)).open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=SOURCE)
            r['pid']=child.pid; write(receipt,r); code=child.wait()
        r.update(exit_code=code,status='completed' if code==0 else 'failed',elapsed_seconds=time.monotonic()-tic)
        if (output/'summary.json').exists():
            summary=read(output/'summary.json');r.update(accepted=summary['accepted_routes'],initialization=summary['initialization']['passed'])
        write(receipt,r);print(json.dumps(r),flush=True);return r
    with ThreadPoolExecutor(max_workers=2) as pool: results=list(pool.map(one,plans))
    write(work/'summary.json',dict(results=results,all_closed=len(results)==len(plans)))


def export():
    value=read(DATA/'registration.json'); out=DATA/'export';out.mkdir(exist_ok=False)
    obs=[];labels=[];meta=[];failures=[]
    for p in value['parent_plan']:
        folder=DATA/'parents'/p['role']/p['parent_id']
        if not (folder/'summary.json').exists():raise RuntimeError('Collection incomplete '+p['parent_id'])
        summary=read(folder/'summary.json')
        if not summary['initialization']['passed']:
            failures.append(dict(parent=p['parent_id'],role=p['role'],reason=summary['initialization'].get('error')));continue
        for r in lines(folder/'observations.jsonl'):
            r['image']=str(folder/r['image']);obs.append(r)
            meta.append(dict(id=r['id'],family_id=p['family_id'],variant=p['variant'],role=p['role'],plan_index=p['index']))
        for r in lines(folder/'supervision.jsonl'):
            for k in ('observation','verification_only','route_config'):r[k]=str(folder/r[k])
            r['routes']=[str(folder/f) for f in r['routes']];labels.append(r)
    for name,rows in [('observations',obs),('supervision',labels),('metadata',meta)]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(out/'manifest.json',dict(registration_sha256=sha(DATA/'registration.json'),
        observations=len(obs),failed_parents=failures,requested_parents=len(value['parent_plan']),
        source_sha256={n:sha(out/n) for n in ('observations.jsonl','supervision.jsonl','metadata.jsonl')},
        reference_kind=value['reference_kind'],failure_denominator_policy='All requested DEV parents retained in final coverage denominator'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','collect','export'])
    p.add_argument('--index',type=int);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=480)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.index)
    elif a.stage=='collect':collect(a.start,a.stop,a.resume)
    else:export()
