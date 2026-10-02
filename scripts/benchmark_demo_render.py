"""Independent on/off rendering audit; never modifies the formal collector.

Both arms collect only TRAIN reach_target_281000 proposal 0, in separate
processes and output roots. These are two additional collection proposals.
Initial/restored observations remain RGB-D. Only get_demo motion recording
temporarily suppresses image capture; every pose/open observation is retained.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np


@contextmanager
def demo_camera_flags(scene, off):
    camera=scene.get_observation_config().front_camera
    before=(camera.rgb,camera.depth,camera.point_cloud,camera.mask)
    if camera.point_cloud or camera.mask:
        raise ValueError('This candidate requires the original RGB-D-only configuration')
    try:
        if off:camera.rgb=camera.depth=False
        yield
    finally:
        camera.rgb,camera.depth,camera.point_cloud,camera.mask=before


def run_arm(output, mode):
    from scripts import observation_collect_multitask as collection
    from rlbench.backend.scene import Scene
    if output.exists():raise FileExistsError('Preserve prior performance arm')
    output.mkdir(parents=True);data=output/'data'
    plan=collection.ensure_registration(data,281000);collection.ensure_source(data)
    original_demo,original_observation=Scene.get_demo,Scene.get_observation
    state=dict(in_demo=False,observations_during_demo=0,rgb_during_demo=0,depth_during_demo=0,
        pose_open_observations=0,demo_calls=0,demo_seconds=0.,flags_restored=True)
    def observe(scene,*args,**kwargs):
        observation=original_observation(scene,*args,**kwargs)
        if state['in_demo']:
            state['observations_during_demo']+=1
            state['rgb_during_demo']+=int(observation.front_rgb is not None)
            state['depth_during_demo']+=int(observation.front_depth is not None)
            state['pose_open_observations']+=int(observation.gripper_pose is not None and observation.gripper_open is not None)
        return observation
    def demo(scene,*args,**kwargs):
        camera=scene.get_observation_config().front_camera;before=(camera.rgb,camera.depth,camera.point_cloud,camera.mask)
        began=time.perf_counter();state['in_demo']=True;state['demo_calls']+=1
        try:
            with demo_camera_flags(scene,mode=='off'):
                return original_demo(scene,*args,**kwargs)
        finally:
            state['in_demo']=False;state['demo_seconds']+=time.perf_counter()-began
            state['flags_restored'] &= (camera.rgb,camera.depth,camera.point_cloud,camera.mask)==before
    Scene.get_demo,Scene.get_observation=demo,observe
    began=time.perf_counter()
    try:
        code=collection.worker(data,plan,plan['parents'][0],max_new_attempts=1)
    finally:
        Scene.get_demo,Scene.get_observation=original_demo,original_observation
    if code!=3:raise RuntimeError('Expected one committed attempt then controlled exit 3; got '+str(code))
    folder=collection.parent_folder(data,plan['parents'][0]);records=collection.completed_records(folder)
    assert len(records)==1 and records[0]['attempt']==0
    assert state['flags_restored'] and state['pose_open_observations']==state['observations_during_demo']
    if mode=='off':assert state['rgb_during_demo']==state['depth_during_demo']==0
    result=dict(mode=mode,worker_seconds=time.perf_counter()-began,worker_exit_code=code,
        original_task_success=records[0]['success'],additional_collection_proposals=1,source_sha256=collection.digest(__file__),
        collector_sha256=collection.digest(collection.__file__),parent_id='reach_target_281000',proposal=0,
        formal_collector_modified=False,performance_override_scope='Scene.get_demo only; no prefix candidate in this probe',**state)
    collection.atomic_json(output/'result.json',result);print(json.dumps(result),flush=True)


def compare(on, off, output):
    from PIL import Image
    from scripts import observation_collect_multitask as collection
    if output.exists():raise FileExistsError('Preserve comparison output')
    summaries=[json.loads((root/'result.json').read_text()) for root in (on,off)]
    arrays=[];world=[];descriptions=[];routes=[]
    for root in (on,off):
        parent=root/'data/TRAIN/parents/reach_target_281000'
        reference=parent/json.loads((parent/'reference_pointer.json').read_text())['directory']
        with np.load(reference/'observation.npz',allow_pickle=False) as archive:item={key:archive[key].copy() for key in archive.files}
        item['rgb']=np.asarray(Image.open(reference/'front.png'));arrays.append(item)
        metadata=json.loads((reference/'reference.json').read_text());world.append(metadata['world']);descriptions.append(metadata['descriptions'])
        record=collection.completed_records(parent)[0]
        if record['success']:
            with np.load(parent/record['route'],allow_pickle=False) as archive:routes.append({key:archive[key].copy() for key in archive.files})
        else:routes.append(None)
    inputs=collection.strict_input_difference(arrays[0],arrays[1]);trajectory={}
    if all(route is not None for route in routes):
        for name in ('gripper_pose','gripper_open'):
            left,right=routes[0][name],routes[1][name];same_shape=left.shape==right.shape
            trajectory[name]=dict(shape_on=list(left.shape),shape_off=list(right.shape),exact_equal=bool(np.array_equal(left,right)),max_absolute_difference=float(np.abs(left-right).max()) if same_shape else None)
    equivalent=(not any(inputs.values()) and world[0]==world[1] and descriptions[0]==descriptions[1] and
        len(trajectory)==2 and all(value['exact_equal'] for value in trajectory.values()) and all(row['original_task_success'] for row in summaries))
    result=dict(arms=summaries,initial_max_differences=inputs,world_exact_equal=world[0]==world[1],language_equal=descriptions[0]==descriptions[1],trajectory_comparison=trajectory,
        single_parent_exact_equivalence=equivalent,worker_wall_speedup=summaries[0]['worker_seconds']/summaries[1]['worker_seconds'],
        get_demo_speedup=summaries[0]['demo_seconds']/summaries[1]['demo_seconds'],additional_collection_proposals=2,
        limitation='One TRAIN parent/proposal performance audit, not a new route-type or method result; no automatic adoption in formal collection',source_sha256=collection.digest(__file__))
    collection.atomic_json(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('on','off'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',nargs=2,type=Path);args=parser.parse_args()
    if args.compare:compare(*args.compare,args.output)
    elif args.mode:run_arm(args.output,args.mode)
    else:parser.error('--mode or --compare required')
