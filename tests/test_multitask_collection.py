"""Filesystem contracts only: no claim of tested simulator reconstruction."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.observation_collect_multitask import (
    TASKS, acquire_lock, atomic_json, completed_records, digest, ensure_registration, ensure_source,
    export_role, parent_folder, pending_slots, registration, strict_input_difference, mechanical_summary,
    demo_render_policy, worker_command,
)


def test_parent_registration_is_disjoint_and_fixed():
    plan=registration(281000)
    assert plan['parents_requested']==144 and plan['attempts_requested']==432
    assert len({p['parent_id'] for p in plan['parents']})==144
    assert len({p['seed'] for p in plan['parents']})==144
    for task in TASKS:
        parents=[p for p in plan['parents'] if p['task']==task]
        assert {role:sum(p['split']==role for p in parents) for role in set(p['split'] for p in parents)}==dict(TRAIN=16,DEV_MODEL=2,DEV_SCORE=2,CALIBRATION=2,TEST_LOCKED=2)


def test_existing_registration_cannot_be_reassigned(tmp_path):
    expected=ensure_registration(tmp_path,281000)
    assert ensure_registration(tmp_path,281000)==expected
    before=(tmp_path/'partition_manifest.json').read_bytes()
    with pytest.raises(ValueError,match='Registration mismatch'):ensure_registration(tmp_path,282000)
    assert (tmp_path/'partition_manifest.json').read_bytes()==before


def test_resume_rejects_collector_source_change(tmp_path):
    ensure_source(tmp_path)
    saved=json.loads((tmp_path/'source_manifest.json').read_text());saved['collector_sha256']='altered'
    atomic_json(tmp_path/'source_manifest.json',saved)
    with pytest.raises(ValueError,match='source mismatch'):ensure_source(tmp_path)


def test_failed_attempt_is_closed_and_success_hash_is_verified(tmp_path):
    atomic_json(tmp_path/'attempts/slot_00/record.json',dict(attempt=0,status='completed',success=False,error='retained'))
    route=tmp_path/'attempts/slot_01/route.npz';route.parent.mkdir(parents=True)
    np.savez_compressed(route,gripper_pose=np.zeros((2,7)),gripper_open=np.ones(2))
    atomic_json(route.parent/'record.json',dict(attempt=1,status='completed',success=True,route=route.relative_to(tmp_path).as_posix(),route_file_sha256=digest(route)))
    assert pending_slots(tmp_path)==[2]
    assert len(completed_records(tmp_path))==2
    route.write_bytes(b'corrupted')
    with pytest.raises(AssertionError):completed_records(tmp_path)


def test_role_join_retains_empty_refs_and_does_not_read_locked(tmp_path):
    plan=registration(281000);train=plan['parents'][0];locked=plan['parents'][22]
    folder=parent_folder(tmp_path,train);ref=folder/'reference_versions/0000'
    atomic_json(ref/'reference.json',dict(descriptions=['one instruction','paraphrase']))
    atomic_json(folder/'reference_pointer.json',dict(directory='reference_versions/0000'))
    atomic_json(folder/'closed.json',dict(status='complete',completed_attempts=3,successes=0))
    for slot in range(3):atomic_json(folder/'attempts'/('slot_%02d'%slot)/'record.json',dict(attempt=slot,status='completed',success=False))
    locked_folder=parent_folder(tmp_path,locked);locked_folder.mkdir(parents=True)
    (locked_folder/'closed.json').write_text('malformed locked file must never be read by TRAIN export')
    assert export_role(tmp_path,'TRAIN',plan)==1
    obs=[json.loads(x) for x in (tmp_path/'TRAIN/observations.jsonl').read_text().splitlines()]
    sup=[json.loads(x) for x in (tmp_path/'TRAIN/supervision.jsonl').read_text().splitlines()]
    assert len(obs)==len(sup)==2
    assert all(set(x)=={'id','parent_id','split','image','instruction'} for x in obs)
    assert all(x['routes']==[] and x['semantic_targets'] is None for x in sup)
    assert not (tmp_path/'TEST_LOCKED/observations.jsonl').exists()


def test_reconstruction_requires_exact_observed_values_and_inventory():
    reference=dict(rgb=np.zeros((2,2,3),np.uint8),depth=np.ones((2,2)),gripper_pose=np.arange(7.))
    assert all(value==0 for value in strict_input_difference(reference,reference).values())
    changed={key:value.copy() for key,value in reference.items()};changed['depth'][0,0]+=1e-9
    assert strict_input_difference(reference,changed)['depth']>0
    changed['rgb'][0,0,0]=1
    assert strict_input_difference(reference,changed)['rgb']==1
    with pytest.raises(ValueError,match='inventory'):strict_input_difference(reference,{'depth':reference['depth']})


def test_live_lock_never_reclaimed(tmp_path):
    lock=tmp_path/'worker.lock';acquire_lock(lock)
    original=lock.read_bytes()
    with pytest.raises(RuntimeError,match='still exists'):acquire_lock(lock,allow_stale=True)
    assert lock.read_bytes()==original and json.loads(original)['pid']==os.getpid()


def test_all_worker_sessions_count_and_unfinished_cost_stays_upper_bound(tmp_path):
    import datetime
    plan=registration(281000);spec=plan['parents'][0];folder=parent_folder(tmp_path,spec)
    atomic_json(folder/'sessions/first.json',dict(status='partial_for_resume_validation',elapsed_seconds=4.,worker_exit_code=3))
    atomic_json(folder/'sessions/second.json',dict(status='setup_failed',elapsed_seconds=7.,worker_exit_code=0))
    began=(datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(seconds=100)).isoformat()
    atomic_json(folder/'sessions/unfinished.json',dict(status='running',elapsed_seconds=None,started_utc=began))
    atomic_json(folder/'closed.json',dict(status='setup_failed',requested_attempts=3,unattempted_slots=3))
    result=mechanical_summary(tmp_path,plan);row=result['parents'][0]
    assert row['finalized_worker_elapsed_seconds']==result['finalized_worker_elapsed_seconds']==11.
    assert row['requested_attempts']==row['unattempted_or_uncommitted_slots']==3
    assert row['unfinished_worker_elapsed_upper_bounds'][0]['elapsed_seconds_upper_bound']>=100.
    assert row['unfinished_worker_elapsed_upper_bounds'][0]['actual_compute_seconds'] is None


def test_whitelist_policy_keeps_push_on_and_restores_after_failure():
    camera=SimpleNamespace(rgb=True,depth=True,point_cloud=False,mask=False)
    scene=SimpleNamespace(get_observation_config=lambda:SimpleNamespace(front_camera=camera))
    for task in TASKS:
        with demo_render_policy(scene,task,'on') as audit:
            assert camera.rgb and camera.depth and not audit['rgbd_suppressed']
        assert audit['flags_restored']
        with pytest.raises(RuntimeError,match='demo failed'):
            with demo_render_policy(scene,task,'validated-five-v1') as audit:
                assert camera.rgb==camera.depth==(task=='push_button')
                assert audit['rgbd_suppressed']==(task!='push_button')
                raise RuntimeError('demo failed')
        assert camera.rgb and camera.depth and audit['flags_restored']
    camera.rgb=False
    with pytest.raises(ValueError,match='RGB-D enabled'):
        with demo_render_policy(scene,'reach_target','validated-five-v1'):pass


def test_camera_policy_and_schedule_cannot_change_on_resume(tmp_path):
    plan=ensure_registration(tmp_path,282000,'validated-five-v1','interleaved-early-dev-v1')
    before=(tmp_path/'partition_manifest.json').read_bytes()
    for policy,schedule in [('on','interleaved-early-dev-v1'),('validated-five-v1','task-major-v1')]:
        with pytest.raises(ValueError,match='Registration mismatch'):
            ensure_registration(tmp_path,282000,policy,schedule)
    assert before==(tmp_path/'partition_manifest.json').read_bytes()
    command=worker_command(tmp_path,plan,plan['parents'][0])
    assert command[command.index('--camera-policy')+1]=='validated-five-v1'
    assert command[command.index('--schedule')+1]=='interleaved-early-dev-v1'


def test_interleaved_registration_preserves_roles_and_new_parent_identity():
    plan=registration(282000,'validated-five-v1','interleaved-early-dev-v1')
    original=registration(281000);same_seed=registration(282000)
    assert {p['parent_id'] for p in plan['parents']}.isdisjoint(p['parent_id'] for p in original['parents'])
    assert {p['seed'] for p in plan['parents']}.isdisjoint(p['seed'] for p in original['parents'])
    assert {p['parent_id']:p for p in plan['parents']}=={p['parent_id']:p for p in same_seed['parents']}
    assert [p['task'] for p in plan['parents'][:6]]==list(TASKS)
    assert [p['parent_index'] for p in plan['parents'][:24:6]]==[0,16,1,17]
    assert [p['split'] for p in plan['parents'][:24:6]]==['TRAIN','DEV_MODEL','TRAIN','DEV_MODEL']
