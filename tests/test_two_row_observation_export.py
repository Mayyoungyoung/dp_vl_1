import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from scripts import export_two_row_observations as exporter
from scripts.collect_observed_two_row_pilot import geometry


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def corpus(tmp_path,monkeypatch):
    selection=json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_prefix28_selection_v1.json').read_text())
    root=tmp_path/'raw';root.mkdir();plans=[]
    for i in range(116):
        role='TRAIN' if i<64 else 'DEV_MODEL' if i<76 else 'TEST_LOCKED'
        plans.append(dict(index=i,parent_id='two_row_reach_%d'%(283200+i),role=role,registered_geometry_1mm_sha256='a'*64))
    registration=dict(parent_plan=plans);write(root/'registration.json',registration)
    manifest=dict(registration_sha256=exporter.digest(root/'registration.json'));write(root/'corpus_manifest.json',manifest)
    monkeypatch.setattr(exporter.collector,'verify_corpus',lambda source:(registration,manifest))
    gate=dict(blocked_parent_ids=[],mechanical_only=True)
    monkeypatch.setattr(exporter.collector,'live_layout_gate',lambda source:gate)
    for i in exporter.SELECTED:
        p=plans[i];c=dict(parent_id=p['parent_id'],index=i,role=p['role'],requested_routes=27,
            initial_observation_saved=False,completed_slots=0,mechanical_files_sha256={},
            actual_geometry_1mm_sha256=None,attempted_lower=0,attempted_upper=0,unattempted_lower=27,unattempted_upper=27)
        write(root/'closures'/('%03d.json'%i),c)
    # If an exporter opens locked outcome data, the deliberately malformed file fails.
    write(root/'parents/TEST_LOCKED/must_not_open/placeholder.json',{})
    (root/'parents/TEST_LOCKED/must_not_open/placeholder.json').write_text('not json')
    return root,selection,plans,gate,tmp_path/'export'


def add_observed_parent(corpus,unknown=False):
    root,selection,plans,_,_=corpus;p=plans[0];parent=p['parent_id'];folder=root/'parents/TRAIN'/parent
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_pilot_v4.json').read_text())
    cfg['split']='TRAIN';p['config']=cfg;p['target_colors']=[dict(name=name) for name in ('red','green','blue')]
    poses=np.c_[np.array([cfg['entry_xyz'],[.07,0,.90 if unknown else 1.0],[.4,0,.90 if unknown else 1.0],cfg['goal_xyz'][1]]),np.tile([0.,0.,0.,1.],(4,1))]
    events=np.ones(4);fields,h24,passed=exporter.route_acceptance(poses[:,:3],np.asarray(cfg['goal_xyz']),1,cfg);assert passed
    (folder/parent).mkdir(parents=True);(folder/'route_configs').mkdir()
    Image.fromarray(np.zeros((224,224,3),np.uint8)).save(folder/parent/'front.png')
    np.savez(folder/parent/'observation.npz',depth=np.ones((224,224)),gripper_pose=poses[0],gripper_open=1.,camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4))
    centers,halves=geometry(cfg);np.savez(folder/parent/'verification_only.npz',obstacle_centers=centers,obstacle_halfsizes=halves,target_centers=cfg['goal_xyz'])
    route=folder/parent/'target1_route0.npz';np.savez(route,gripper_pose=poses,gripper_open=events,xyz_24=h24)
    write(folder/'route_configs'/(parent+'.json'),cfg)
    obs=[dict(id=parent+'_target%d'%t,parent_id=parent,split='TRAIN',image=parent+'/front.png',
        instruction='Move the gripper to touch the %s sphere while avoiding the gray posts.'%p['target_colors'][t]['name']) for t in (0,1)]
    (folder/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in obs))
    strict=dict(passed=True,world=dict(max_abs=0,rgb_max_difference=0,same_object_inventory=True,global_inventory_equal=True),
        observation={key:True for key in ('front_rgb_equal','front_depth_equal','gripper_pose_equal','gripper_open_equal','front_camera_intrinsics_equal','front_camera_extrinsics_equal')})
    record=dict(parent_id=parent,input_id=parent+'_target1',attempt=0,success=True,strict_restore=strict,
        actual_route_type=fields['actual_route_type'],trajectory=dict(file=route.name,sha256=exporter.digest(route),
            pose_array_sha256=exporter.collector.physical.legacy.array_hash(poses),samples=4))
    (folder/'attempts.jsonl').write_text(json.dumps(record)+'\n')
    # No supervision file: interrupted target completion must not erase its recorded positive.
    files={path.relative_to(folder).as_posix():exporter.digest(path) for path in folder.rglob('*') if path.is_file()}
    write(folder/'artifact_hashes.json',files)
    closure=json.loads((root/'closures/000.json').read_text());closure.update(initial_observation_saved=True,completed_slots=1,
        actual_geometry_1mm_sha256='a'*64,mechanical_files_sha256={'artifact_hashes.json':exporter.digest(folder/'artifact_hashes.json')})
    write(root/'closures/000.json',closure);return folder


def test_fixed_failures_preserved_and_no_raw_locked_reads(corpus):
    root,selection,_,_,out=corpus;m=exporter.build(root,selection,out)
    assert m['requested_parents']==28 and m['requested_routes']==756 and m['actual_inputs']==0
    assert m['unobserved_requested_inputs']==84
    assert len(json.loads((out/'parent_inventory.json').read_text()))==28
    exporter.verify_export(out)


@pytest.mark.parametrize('unknown',[False,True])
def test_real_inputs_missing_labels_and_unknown_positive_preserved(corpus,unknown):
    add_observed_parent(corpus,unknown);root,selection,_,_,out=corpus
    m=exporter.build(root,selection,out);assert m['actual_inputs']==2 and m['positive_references']==1
    assert m['unobserved_requested_inputs']==82
    labels=[json.loads(r) for r in (out/'supervision.jsonl').read_text().splitlines()]
    assert not labels[0]['routes'] and len(labels[1]['routes'])==1
    assert all(not r['original_supervision_present'] for r in labels)
    assert (labels[1]['route_types']==[None])==unknown
    obs=[json.loads(r) for r in (out/'observations.jsonl').read_text().splitlines()]
    assert all(set(r)==exporter.INPUT_KEYS for r in obs)


def test_unclosed_parent_not_replaced_and_no_extra_train_allowed(corpus):
    root,selection,_,_,out=corpus;(root/'closures/075.json').unlink()
    with pytest.raises(ValueError,match='not fully closed'):exporter.build(root,selection,out)
    changed=copy.deepcopy(selection);changed['selected_indices'][-1]=76
    with pytest.raises(ValueError,match='fixed16 TRAIN'):exporter.closed_prefix(root,changed)
    assert not out.exists()


def test_live_duplicate_or_post_export_source_corruption_refuses_use(corpus):
    root,selection,plans,gate,out=corpus;folder=add_observed_parent(corpus)
    exporter.build(root,selection,out);gate['blocked_parent_ids']=[plans[0]['parent_id']]
    with pytest.raises(ValueError,match='Current model-use gate blocked'):exporter.verify_export(out)
    gate['blocked_parent_ids']=[];(folder/'attempts.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='Closed source changed'):exporter.verify_export(out)


def test_success_reference_restore_cannot_be_weakened():
    assert not exporter.strict_restore(dict(strict_restore=dict(passed=True,world=dict(max_abs=.000001))))
