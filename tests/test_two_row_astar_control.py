import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import evaluate_two_row_astar_v2 as control
from scripts.collect_observed_two_row_pilot import geometry


def make_fixture(root):
    root.mkdir(exist_ok=True)
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_pilot_v4.json').read_text())
    parent='p';folder=root/parent;folder.mkdir()
    Image.fromarray(np.zeros((4,4,3),np.uint8)).save(folder/'front.png')
    current=np.r_[config['entry_xyz'],0.,0.,0.,1.]
    np.savez(folder/'observation.npz',depth=np.ones((4,4)),gripper_pose=current,gripper_open=1.,camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4))
    centers,halves=geometry(config)
    np.savez(folder/'verification.npz',obstacle_centers=centers,obstacle_halfsizes=halves)
    (folder/'route_config.json').write_text(json.dumps(config))
    raw=np.asarray([config['entry_xyz'],[.07,0,1.0],[.4,0,1.0],config['goal_xyz'][1]])
    parameter=np.linspace(0,len(raw)-1,24)
    path=np.stack([np.interp(parameter,np.arange(len(raw)),raw[:,i]) for i in range(3)],axis=-1)
    paths=np.full((4,24,3),np.nan);paths[:2]=path
    raw_pool=[raw.copy(),raw.copy(),None,None]
    row=dict(id='p_target1',parent_id=parent,split='DEV_MODEL',image=str(folder/'front.png'),instruction='known')
    label=dict(id=row['id'],parent_id=parent,split='DEV_MODEL',observation=str(folder/'observation.npz'),
        verification_only=str(folder/'verification.npz'),route_config=str(folder/'route_config.json'),routes=[],route_types=[],
        semantic_targets=dict(centers=config['goal_xyz'],target_index=1,tolerance=.03))
    (root/'supervision.jsonl').write_text(json.dumps(label)+'\n')
    record=dict(submitted_candidate_budget=4,raw_complete_paths=2,failed_slots=2,localization=dict(status='predicted'),
        attempts=[dict(slot=i,status='path_found' if i<2 else 'open_set_exhausted',complete_raw_paths_emitted=int(i<2),
            exact_grid_path_duplicate=i==1) for i in range(4)])
    calls=[]
    def plan(rgb,xyz,valid,depth,intrinsics,camera,current,instruction,model,prior):
        calls.append(instruction)
        assert set(model)=={'prototypes'} and set(prior)=={'lower','upper'}
        assert instruction==row['instruction']
        return paths.copy(),raw_pool,record
    def grid(rgb,depth,intrinsics,camera):
        return rgb/255.,np.zeros(depth.shape+(3,)),np.ones(depth.shape,bool)
    planner=SimpleNamespace(plan_observation=plan,prototype=SimpleNamespace(observed_grid=grid))
    manifest=dict(source_files_sha256={str(p):control.digest(p) for p in folder.iterdir()})
    return row,label,manifest,planner,calls


def test_four_slots_failed_nan_and_duplicate_are_preserved_before_labels(tmp_path,monkeypatch):
    data=tmp_path/'data';row,label,manifest,planner,calls=make_fixture(data);out=tmp_path/'request'
    original=control.label_for
    def guarded(data,row):
        seal=json.loads((out/'generation_seal.json').read_text())
        assert seal['prediction_sha256']==control.digest(out/'predictions.npz') and not seal['evaluation_labels_opened']
        assert len(calls)==1
        return original(data,row)
    monkeypatch.setattr(control,'label_for',guarded)
    result,paths,events=control.one_request(data,row,manifest,{'prototypes':{}},{'lower':[],'upper':[]},planner,out)
    assert result['metrics']['candidates']==4 and np.isnan(paths[2:]).all()
    assert np.array_equal(paths[0],paths[1]) and (events==1).all()
    metric=control.aggregate([result])
    assert metric['failed_candidate_slots']==2 and metric['exact_grid_duplicate_slots']==1
    assert metric['requested_dev_inputs']==36 and metric['unobserved_requested_dev_inputs']==35
    assert metric['submitted_candidate_budget']==4 and metric['unattempted_missing_input_slots']==140
    assert metric['examples_without_reference']==1 and metric['known_reference_coverage_evaluation_examples']==0
    assert metric['KnownReferenceTypeCoverageAtK'] is None


def test_dev_target_changes_cannot_change_same_observation_planner_output(tmp_path):
    data=tmp_path/'data';row,label,manifest,planner,_=make_fixture(data)
    first,paths,_=control.one_request(data,row,manifest,{'prototypes':{}},{'lower':[],'upper':[]},planner,tmp_path/'first')
    label['semantic_targets']['target_index']=0
    (data/'supervision.jsonl').write_text(json.dumps(label)+'\n')
    second,again,_=control.one_request(data,row,manifest,{'prototypes':{}},{'lower':[],'upper':[]},planner,tmp_path/'second')
    np.testing.assert_array_equal(paths,again)
    assert first['metrics']['semantic_goal_accuracy']==.5 and second['metrics']['semantic_goal_accuracy']==0.


def test_budget_rejects_extra_slot_or_erased_failure():
    paths=np.full((4,24,3),np.nan);raw=[None]*4
    record=dict(submitted_candidate_budget=4,raw_complete_paths=0,failed_slots=4,
        attempts=[dict(slot=i,complete_raw_paths_emitted=0) for i in range(4)])
    control.validate_pool(paths,raw,record)
    with pytest.raises(ValueError,match='four bounded'):control.validate_pool(np.zeros((5,24,3)),[None]*5,record)
    paths[0]=0
    with pytest.raises(ValueError,match='remain NaN'):control.validate_pool(paths,raw,record)
    paths[:]=np.nan;record['attempts'][2]['complete_raw_paths_emitted']=1
    with pytest.raises(ValueError,match='remain NaN'):control.validate_pool(paths,raw,record)


def test_fit_only_train_positives_and_no_semantic_or_mode_fields(tmp_path,monkeypatch):
    data=tmp_path/'data';row,label,manifest,_,_=make_fixture(data)
    train=dict(row,id='train',parent_id='train_parent',split='TRAIN')
    route=data/'p/positive.npz';np.savez(route,gripper_pose=np.zeros((2,7)),gripper_open=np.ones(2))
    manifest['source_files_sha256'][str(route)]=control.digest(route)
    def selected_label(data,r):
        assert r==train
        return dict(label,id='train',parent_id='train_parent',split='TRAIN',routes=[str(route)],route_types=['must_not_pass'],semantic_targets='must_not_pass')
    monkeypatch.setattr(control,'label_for',selected_label)
    def check(rows,labels):
        assert rows==[train] and set(labels)=={'train'} and set(labels['train'])=={'routes','observation'}
    def fit(data,rows,labels):
        check(rows,labels);return dict(training_source_sha256={str(route):control.digest(route)}),{'train_parent'}
    def workspace(data,rows,labels):
        check(rows,labels);return dict(source_sha256={str(route):control.digest(route)})
    planner=SimpleNamespace(prototype=SimpleNamespace(fit_prototypes=fit),v1=SimpleNamespace(fit_workspace=workspace))
    _,_,receipt=control.fit_train(data,[train,row],manifest,planner)
    assert receipt['positive_reference_occurrences']==1 and not receipt['semantic_targets_or_route_types_passed_to_fitters']


def test_unknown_language_keeps_four_failure_slots_in_actual_frozen_planner():
    pytest.importorskip('scipy')
    planner=control.dependencies()
    paths,raw,record=planner.plan_observation(None,None,None,None,None,None,np.r_[0.,0.,0.,0.,0.,0.,1.,1.],
        'unseen exact instruction',{'prototypes':{}},None)
    control.validate_pool(paths,raw,record)
    assert record['localization']['status']=='unsupported_exact_instruction' and np.isnan(paths).all()


def test_source_guard_and_input_label_whitelist(tmp_path,monkeypatch):
    data=tmp_path/'data';row,label,manifest,planner,_=make_fixture(data)
    with pytest.raises(ValueError,match='Strict DEV'):
        control.load_current(dict(row,target_xyz=[1,2,3]),manifest,{},planner)
    (data/'p/front.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Closed source changed'):control.load_current(row,manifest,{},planner)


@pytest.mark.parametrize('name',list(control.FROZEN_SOURCES))
def test_only_registered_lf_and_crlf_source_bytes_are_accepted(tmp_path,name):
    source=Path(control.__file__).with_name(name).read_bytes().replace(b'\r\n',b'\n')
    path=tmp_path/name
    for form,content in [('LF',source),('CRLF',source.replace(b'\n',b'\r\n'))]:
        path.write_bytes(content);receipt=control.verify_frozen_source(path)
        assert receipt['actual_sha256']==control.digest(path) and receipt['exact_byte_form']==form
        assert receipt['canonical_lf_sha256']==control.FROZEN_SOURCES[name]


def test_source_guard_rejects_mixed_newlines_whitespace_and_changed_code(tmp_path):
    name='observation_multiroute_astar_v2.py'
    source=Path(control.__file__).with_name(name).read_bytes().replace(b'\r\n',b'\n')
    path=tmp_path/name
    for content in [source.replace(b'\n',b'\r\n',1),source+b' ',source.replace(b'2.5mm',b'3.5mm',1)]:
        assert content!=source
        path.write_bytes(content)
        with pytest.raises(ValueError,match='two exact LF/CRLF forms'):control.verify_frozen_source(path)
