import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import diagnose_two_row_static_contacts as diagnostic


def make_source(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'manifest.json').write_text(json.dumps({'config':{'protocol':'two_row_endpoint_ik_24_v1','geometry':{}}}))
    rows=[dict(query_id=i,status='configuration_returned_diagnostic_only',ignore_collisions=True,
        endpoint_collision='arm_environment',endpoint_joint_readback_max_abs=0,returned_configs=[[0]*7]) for i in range(24)]
    (source/'endpoint_queries.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    (source/'common_initial.json').write_text(json.dumps({'state':{},'inventory':[]}))
    rgb=np.zeros((2,2,3),dtype=np.uint8);Image.fromarray(rgb).save(source/'common_initial.png')
    np.savez(source/'common_initial.npz',rgb=rgb,depth=np.ones((2,2)),gripper_pose=np.r_[np.zeros(6),1],
        gripper_open=1,camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4))
    return dict(protocol='two_row_static_collision_six_v1',query_ids=diagnostic.QUERY_IDS.copy(),source_dataset=str(source),
        source_files_sha256={name:diagnostic.digest(source/name) for name in diagnostic.SOURCE_FILES})


def test_only_six_frozen_collision_queries(tmp_path):
    loaded=diagnostic.load_frozen(make_source(tmp_path))
    assert [r['query_id'] for r in loaded['queries']]==[5,7,13,15,21,23]


def test_source_corruption_rejected(tmp_path):
    config=make_source(tmp_path)
    (Path(config['source_dataset'])/'endpoint_queries.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='SHA'):diagnostic.load_frozen(config)


def test_unregistered_configuration_ids_rejected(tmp_path):
    config=make_source(tmp_path);config['query_ids'][0]=4
    with pytest.raises(ValueError):diagnostic.load_frozen(config)


def test_assets_must_match_pinned_archive(tmp_path):
    package=tmp_path/'installed';(package/'task_ttms').mkdir(parents=True)
    archive=tmp_path/'archive.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ['task_design.ttt','task_ttms/reach_target.ttm']:
            content=('fixed model '+name).encode();(package/name).write_bytes(content)
            info=tarfile.TarInfo('RLBench-fixed/rlbench/'+name);info.size=len(content);tar.addfile(info,io.BytesIO(content))
    config=dict(rlbench_archive=str(archive),rlbench_archive_sha256=diagnostic.digest(archive))
    assert len(diagnostic.verify_assets(config,package))==2
    (package/'task_design.ttt').write_bytes(b'changed model')
    with pytest.raises(ValueError,match='differs'):diagnostic.verify_assets(config,package)


def test_new_static_world_comparison_is_exact():
    a=dict(state={'joint':{'value':[1.]}},inventory=[('joint',1,1)])
    b=dict(state={'joint':{'value':[1.+1e-12]}},inventory=[['joint',1,1]])
    assert diagnostic.world_difference(a,a)['exact']
    assert not diagnostic.world_difference(a,b)['exact']
    assert diagnostic.world_difference(a,b)['max_abs']>0


def test_missing_world_field_or_nonfinite_blocks_restore():
    a=dict(state={'joint':{'value':[1.]}},inventory=[])
    for state in ({},{'joint':{'value':[float('nan')]}}):
        assert not diagnostic.world_difference(a,dict(state=state,inventory=[]))['exact']


def test_collision_group_does_not_conflate_robot_and_posts():
    assert diagnostic.collision_body_group('post',{'post'},{'link'},{'ball'})=='registered_post'
    assert diagnostic.collision_body_group('link',{'post'},{'link'},{'ball'})=='robot_body'
    assert diagnostic.collision_body_group('ball',{'post'},{'link'},{'ball'})=='task_body'
    assert diagnostic.collision_body_group('table',{'post'},{'link'},{'ball'})=='other_world_body'


def test_collection_collision_and_link_pair_are_separate():
    class Shape:
        def __init__(self,name,handle):self.name=name;self.handle=handle
        def get_name(self):return self.name
        def get_handle(self):return self.handle
        def is_collidable(self):return True
        def check_collision(self,body):return self.name=='link6' and body.name=='post'
    link=Shape('link6',1);post=Shape('post',2);table=Shape('table',3)
    arm=SimpleNamespace(check_arm_collision=lambda body:body.name=='post')
    rows=diagnostic.collision_matrix(arm,[link],[post,table],{'post'},{'link6'},set())
    assert rows[0]['arm_collection_collision'] and rows[0]['intersecting_robot_shapes']==['link6']
    assert not rows[1]['arm_collection_collision'] and rows[1]['intersecting_robot_shapes']==[]


def test_guard_blocks_ik_routes_starts_but_services_ui_and_restores():
    names=['get_path','get_linear_path','get_nonlinear_path','solve_ik','solve_ik_via_sampling','solve_ik_via_jacobian']
    arm=SimpleNamespace(**{name:lambda:None for name in names});original=arm.get_path
    ui=[];sim=SimpleNamespace(**{name:lambda:None for name in ['simGetConfigForTipPose','simCheckIkGroup','generateIkPath','simStartSimulation']},
        simExtStep=lambda value:ui.append(value))
    count=dict(ik_attempts=0,path_attempts=0,simulation_start_attempts=0,ui_updates=0)
    with diagnostic.zero_motion_guard(arm,sim,count):
        for call in [arm.solve_ik_via_sampling,arm.get_path,sim.simStartSimulation]:
            with pytest.raises(RuntimeError):call()
        sim.simExtStep();sim.simExtStep(True)
    assert ui==[False,False]
    assert count==dict(ik_attempts=1,path_attempts=1,simulation_start_attempts=1,ui_updates=2)
    assert arm.get_path is original


def fake_hierarchy(parents,types):
    return SimpleNamespace(simGetObjectType=lambda handle:types[handle],
        lib=SimpleNamespace(simGetObjectParent=lambda handle:parents[handle]))


def test_hierarchy_handles_root_sentinel_and_parent_order():
    inventory=[('grandchild',3,0),('root',1,0),('child',2,1),('other_root',4,2)]
    sim=fake_hierarchy({1:-1,2:1,3:2,4:-1},{1:0,2:1,3:0,4:2})
    assert diagnostic.hierarchy_depths(inventory,sim)=={1:0,2:1,3:2,4:0}


def test_hierarchy_rejects_parent_outside_inventory():
    with pytest.raises(RuntimeError,match='outside'):
        diagnostic.hierarchy_depths([('child',1,0)],fake_hierarchy({1:2},{1:0}))


def test_hierarchy_rejects_cycles_and_duplicate_handles():
    with pytest.raises(RuntimeError,match='cyclic'):
        diagnostic.hierarchy_depths([('a',1,0),('b',2,0)],fake_hierarchy({1:2,2:1},{1:0,2:0}))
    with pytest.raises(RuntimeError,match='duplicate'):
        diagnostic.hierarchy_depths([('a',1,0),('b',1,0)],fake_hierarchy({1:-1},{1:0}))


def test_hierarchy_propagates_invalid_handle_and_rejects_changed_type():
    def invalid(handle):raise RuntimeError('invalid handle')
    sim=fake_hierarchy({1:-1},{1:0});sim.simGetObjectType=invalid
    with pytest.raises(RuntimeError,match='invalid handle'):
        diagnostic.hierarchy_depths([('root',1,0)],sim)
    with pytest.raises(RuntimeError,match='type changed'):
        diagnostic.hierarchy_depths([('root',1,0)],fake_hierarchy({1:-1},{1:1}))
