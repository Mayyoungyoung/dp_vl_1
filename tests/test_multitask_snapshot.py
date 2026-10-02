import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from routeset.multitask_fingerprints import fingerprint
from scripts.observation_collect_multitask import registration
from scripts.snapshot_multitask_observations import CURRENT_KEYS,build,child_path,digest,select_prefix,write_json


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,value)


def make_corpus(tmp_path,selected=None):
    source=tmp_path/'source';other=tmp_path/'legacy';source.mkdir();other.mkdir()
    plan=registration(282000,'validated-five-v1','interleaved-early-dev-v1')
    write_json(source/'partition_manifest.json',plan);write_json(other/'partition_manifest.json',dict(parents=[]))
    manifest=dict(collector_sha256='collector',restore_helper_sha256='restore',fingerprint_helper_sha256='fingerprint')
    write_json(source/'source_manifest.json',manifest)
    selected=select_prefix(plan) if selected is None else selected
    for number,spec in enumerate(selected):
        folder=source/spec['split']/'parents'/spec['parent_id']
        save_json(folder/'sessions/one.json',dict(status='complete',elapsed_seconds=1.,worker_exit_code=0))
        if number==0:
            save_json(folder/'closed.json',dict(status='setup_failed',requested_attempts=3,completed_attempts=0,
                successes=0,failed_attempts=0,unattempted_slots=3,total_proposal_executions=0))
            continue
        reference=folder/'reference_versions/0000';reference.mkdir(parents=True)
        position=.1+spec['parent_index']*.01
        world={'object':dict(pose=[position,0.,1.,0.,0.,0.,1.],color=[.2,0.,0.],velocity=[0.]*6)}
        current=dict(depth=np.ones((3,3),np.float32),gripper_pose=np.array([0.,0.,1.,0.,0.,0.,1.]),
            gripper_open=np.array(1.),camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4))
        rgb=np.full((3,3,3),spec['parent_index']+10,np.uint8)
        Image.fromarray(rgb).save(reference/'front.png');np.savez_compressed(reference/'observation.npz',**current)
        descriptions=['instruction','paraphrase']+(['third wording'] if number==1 else [])
        write_json(reference/'reference.json',dict(world=world,descriptions=descriptions))
        save_json(folder/'reference_pointer.json',dict(directory='reference_versions/0000',
            image_sha256=digest(reference/'front.png'),observation_sha256=digest(reference/'observation.npz'),reference_sha256=digest(reference/'reference.json')))
        mechanical=fingerprint(world,dict(current,rgb=rgb),spec);mechanical['rgb_file_sha256']=digest(reference/'front.png')
        save_json(folder/'mechanical_fingerprint.json',mechanical)
        count=0 if number==1 else 1
        save_json(folder/'closed.json',dict(status='complete',requested_attempts=3,completed_attempts=3,
            successes=count,failed_attempts=3-count,total_proposal_executions=3,interrupted_proposal_executions=0))
        for attempt in range(3):
            row=dict(parent_id=spec['parent_id'],status='completed',attempt=attempt,success=bool(count and attempt==0),
                motion_camera_policy='validated-five-v1',demo_render_audit=None,**manifest,
                restore=dict(max_abs=0.,same_object_inventory=True,language_equal=True,
                    observed_field_max_difference={key:0. for key in CURRENT_KEYS|{'rgb'}}))
            directory=folder/'attempts'/('slot_%02d'%attempt);directory.mkdir(parents=True)
            if row['success']:
                pose=np.stack([current['gripper_pose'],current['gripper_pose']+np.array([.02,0.,0.,0.,0.,0.,0.])]);opened=np.array([1.,0.])
                route=directory/'route.npz';np.savez_compressed(route,gripper_pose=pose,gripper_open=opened)
                row.update(route=route.relative_to(folder).as_posix(),route_file_sha256=digest(route),steps=2,
                    trajectory_sha256=hashlib.sha256(pose.tobytes()).hexdigest(),event_sha256=hashlib.sha256(opened.tobytes()).hexdigest(),event_transition_indices=[0],
                    demo_render_audit=dict(policy='validated-five-v1',flags_restored=True,initial_and_restore_rgbd_on=True,rgbd_suppressed=spec['task']!='push_button'))
            else:row['error']='retained genuine collection failure'
            write_json(directory/'record.json',row)
    # Raw non-development contents must not be opened by the exporter.
    locked=next(row for row in plan['parents'] if row['split']=='TEST_LOCKED')
    locked_folder=source/'TEST_LOCKED'/'parents'/locked['parent_id'];locked_folder.mkdir(parents=True)
    for name in ('reference.json','front.png','observation.npz','closed.json'):(locked_folder/name).write_bytes(b'forbidden malformed raw data')
    return source,other,selected


@pytest.fixture
def corpus(tmp_path):
    return make_corpus(tmp_path)


def test_snapshot_retains_variable_language_zero_refs_and_failure_denominators(corpus,tmp_path):
    source,other,_=corpus;out=tmp_path/'snapshot'
    result=build(source,[other],out)
    assert result['requested_parents']==24 and result['requested_attempts']==72
    assert result['actual_attempt_records']==69 and result['unattempted_slots']==3
    assert result['setup_failed_parents']==1 and result['parents_with_zero_reference']==1
    assert result['successful_reference_routes']==22 and result['observations']==47
    rows=[json.loads(line) for line in (out/'observations.jsonl').read_text().splitlines()]
    labels=[json.loads(line) for line in (out/'supervision.jsonl').read_text().splitlines()]
    assert all(set(row)=={'id','parent_id','split','image','instruction'} for row in rows)
    assert all(row['semantic_targets'] is None for row in labels)
    assert sum(not row['routes'] for row in labels)==3
    assert {row['split'] for row in rows}=={'TRAIN','DEV_MODEL'}


def test_waits_for_registered_prefix_instead_of_substituting_successes(corpus,tmp_path):
    source,other,selected=corpus;spec=selected[-1]
    (source/spec['split']/'parents'/spec['parent_id']/'closed.json').rename(source/'temporarily_not_closed.json')
    with pytest.raises(ValueError,match='not fully closed'):build(source,[other],tmp_path/'snapshot')
    assert not (tmp_path/'snapshot').exists()


def test_mandatory_cross_batch_gate_blocks_whole_snapshot_without_relabelling(corpus,tmp_path):
    source,other,selected=corpus;spec=next(row for row in selected if row['split']=='DEV_MODEL')
    original=source/spec['split']/'parents'/spec['parent_id']/'mechanical_fingerprint.json'
    duplicate=json.loads(original.read_text());duplicate.update(parent_id='legacy_duplicate',split='TRAIN')
    write_json(other/'partition_manifest.json',dict(parents=[dict(parent_id='legacy_duplicate',task=spec['task'],split='TRAIN')]))
    save_json(other/'TRAIN/parents/legacy_duplicate/mechanical_fingerprint.json',duplicate)
    before=original.read_bytes()
    with pytest.raises(ValueError,match='Layout gate blocks'):build(source,[other],tmp_path/'snapshot')
    assert not (tmp_path/'snapshot').exists() and (tmp_path/'snapshot.blocked.json').exists()
    assert original.read_bytes()==before


def test_successful_route_hash_corruption_is_rejected(corpus,tmp_path):
    source,other,_=corpus;path=next(source.glob('TRAIN/parents/*/attempts/*/route.npz'))
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='file hash changed'):build(source,[other],tmp_path/'snapshot')


def test_unfinalized_closed_worker_is_not_exported(corpus,tmp_path):
    source,other,selected=corpus;spec=selected[1]
    path=source/spec['split']/'parents'/spec['parent_id']/'worker.lock';path.write_text('still shutting down')
    with pytest.raises(ValueError,match='worker lock'):build(source,[other],tmp_path/'snapshot')


def test_parent_path_rejects_traversal_and_absolute_escape(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();outside=tmp_path/'outside';outside.mkdir()
    assert child_path(parent,'reference/data.npz')==parent/'reference/data.npz'
    for value in ('../outside/data.npz',str(outside/'data.npz')):
        with pytest.raises(ValueError,match='escapes registered parent'):child_path(parent,value)


def test_parent_path_rejects_symbolic_link_escape(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();outside=tmp_path/'outside';outside.mkdir()
    try:(parent/'link').symlink_to(outside,target_is_directory=True)
    except (OSError,NotImplementedError):pytest.skip('Local account cannot create symlinks; run this case on Linux')
    with pytest.raises(ValueError,match='escapes registered parent'):child_path(parent,'link/data.npz')
