import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import analyze_two_row_formal_train as analysis

formal=analysis.formal
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def registered(tmp_path_factory):
    root=tmp_path_factory.mktemp('registered')/'corpus'
    formal.prepare(ROOT/'configs/observed_two_row_formal116_registered_v1.json',root)
    return root


@pytest.fixture
def corpus(registered,tmp_path):
    root=tmp_path/'corpus';shutil.copytree(registered,root);return root


def close_all_failed(corpus):
    value,_=formal.verify_corpus(corpus)
    for p in value['parent_plan'][:16]:
        data=corpus/'parents'/'TRAIN'/p['parent_id']
        closure=formal.mechanical_closure(data,p,{},1.,1)
        formal.atomic_write(corpus/'closures'/('%03d.json'%p['index']),closure)
    return value


def test_missing_prefix_closure_refuses_before_raw_access(corpus,monkeypatch,tmp_path):
    monkeypatch.setattr(analysis.TrainReader,'__init__',lambda *a:(_ for _ in ()).throw(AssertionError('opened raw early')))
    with pytest.raises(ValueError,match='entire fixed TRAIN prefix'):
        analysis.analyze(corpus,tmp_path/'out',plots=False)
    assert not (tmp_path/'out').exists()


def test_all_failed_parents_preserve_432_slots_and_48_conditions_without_nontrain_raw(corpus,tmp_path,monkeypatch):
    value=close_all_failed(corpus)
    private=corpus/'parents'/'TEST_LOCKED'/'two_row_reach_283300';private.mkdir(parents=True)
    (private/'attempts.jsonl').write_text('do not read')
    original=Path.open
    def guarded(path,*args,**kwargs):
        if 'TEST_LOCKED' in path.parts:raise AssertionError('locked raw opened')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=analysis.analyze(corpus,tmp_path/'out',plots=False)
    assert result['slot_status_counts']=={'unattempted':432}
    assert len(result['conditions'])==48 and result['requested_parents']==16
    assert result['accepted_per_requested_slot']==0 and result['unattempted_lower']==432
    assert len(json.loads((tmp_path/'out/all_432_slots.json').read_text()))==432
    assert not result['dev_or_higher_raw_opened']


def test_parent_or_artifact_traversal_rejected(tmp_path):
    reader=analysis.TrainReader(tmp_path/'parent')
    with pytest.raises(ValueError,match='escaped'):reader.path('../other/route.npz')


def test_symlink_escape_rejected(tmp_path):
    root=tmp_path/'parent';outside=tmp_path/'private';root.mkdir();outside.mkdir()
    try:(root/'link').symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('symlink privilege unavailable on host')
    with pytest.raises(ValueError,match='escaped'):analysis.TrainReader(root).path('link/route.npz')


def test_changed_index_digest_rejected(tmp_path):
    root=tmp_path/'parent';root.mkdir();(root/'x').write_bytes(b'original')
    reader=analysis.TrainReader(root);expected=analysis.batch.digest(root/'x')
    (root/'x').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA mismatch'):reader.checked('x',expected)


def test_issued_partial_and_uncertain_tail_not_called_failed_or_unattempted():
    parent='two_row_reach_283200'
    ledger=[dict(event='started',parent_id=parent,input_id=parent+'_target0',attempt=0)]
    slots=analysis.indexed_slots(parent,[],ledger,partial_ledger=True)
    assert len(slots)==27 and slots[0]['status']=='interrupted'
    assert slots[1]['status']=='unattempted_or_unrecorded'
    assert sum(s['status']=='unattempted' for s in slots)==25


def test_duplicate_or_unissued_record_rejected():
    parent='two_row_reach_283200';record=dict(parent_id=parent,input_id=parent+'_target0',attempt=0,success=False)
    with pytest.raises(ValueError,match='unissued'):analysis.indexed_slots(parent,[record],[])
    ledger=[dict(record,event='started')]
    with pytest.raises(ValueError,match='duplicate'):analysis.indexed_slots(parent,[record,record],ledger)


def test_unknown_reference_remains_positive_but_is_not_a_new_known_type():
    types=[['middle','middle'],['middle','middle'],None]
    slots=[dict(status='accepted',record=dict(success=True,actual_route_type=t)) for t in types]
    slots += [dict(status='unattempted',record=None) for _ in range(6)]
    result=analysis.condition_summary(slots)
    assert result['valid_references']==3 and result['distinct_known_types']==1
    assert result['unknown_valid_references']==1 and result['known_duplicate_references']==1
    assert result['total_solution_count'] is None


def valid_trace(tmp_path):
    config=json.loads((ROOT/'configs/observed_two_row_pilot_v4.json').read_text())
    goals=np.asarray(config['goal_xyz']);xyz=np.vstack([config['entry_xyz'],analysis.pilot.proposed_waypoints(goals[1],('negative_y','negative_y'),config)])
    fields,h24,passed=analysis.physical.route_acceptance(xyz,goals,1,config);assert passed
    parent='two_row_reach_283200';folder=tmp_path/parent;folder.mkdir()
    trace=folder/'route.npz';pose=np.c_[xyz,np.tile([0.,0.,0.,1.],(len(xyz),1))]
    np.savez_compressed(trace,gripper_pose=pose,gripper_open=np.ones(len(xyz)),xyz_24=h24)
    segment=dict(segment=0,simulated_steps=len(xyz)-1,goal_supervision_only=goals[1].tolist(),get_path_calls=1,
        planning_seconds=.1,simulation_seconds=.2,planning_status='success',simulation_status='success')
    record=dict(success=True,actual_route_type=list(fields['actual_route_type']),strict_restore={'passed':True},planning_segments=[segment],
        trajectory=dict(file='route.npz',sha256=analysis.batch.digest(trace),samples=len(xyz)))
    slot=dict(parent_id=parent,input_id=parent+'_target1',target=1,attempt=0,status='accepted',issued=True,ledger_completed=True,record=record)
    return dict(parent_id=parent,config=config),slot,goals,trace


def test_rechecks_raw_h24_and_actual_type_without_using_proposal_id(tmp_path):
    plan,slot,goals,_=valid_trace(tmp_path)
    slot['record']['proposed_type_supervision_only']=['positive_y','positive_y']
    row,xyz=analysis.trace_diagnostics(analysis.TrainReader(tmp_path),plan,slot,goals)
    assert row['actual_accepted_type']==['negative_y','negative_y']
    assert row['diagnostic_tip_endpoint_h24_passed'] and len(row['actual_row_crossings'])>0


def test_wrong_saved_h24_and_acceptance_type_rejected(tmp_path):
    plan,slot,goals,trace=valid_trace(tmp_path)
    changed=copy.deepcopy(slot);changed['record']['actual_route_type']=['middle','middle']
    with pytest.raises(ValueError,match='unchanged raw/H24/type'):
        analysis.trace_diagnostics(analysis.TrainReader(tmp_path),plan,changed,goals)
    with np.load(trace) as values:arrays={k:values[k] for k in values.files}
    arrays['xyz_24'][5,2]+=.001;np.savez_compressed(trace,**arrays)
    slot['record']['trajectory']['sha256']=analysis.batch.digest(trace)
    with pytest.raises(ValueError,match='stored H24'):
        analysis.trace_diagnostics(analysis.TrainReader(tmp_path),plan,slot,goals)


def test_failed_whole_robot_path_not_promoted_when_tip_proxy_passes(tmp_path):
    plan,slot,goals,_=valid_trace(tmp_path)
    slot['status']='failed';slot['record']['success']=False;slot['record']['error']='arm_environment'
    row,_=analysis.trace_diagnostics(analysis.TrainReader(tmp_path),plan,slot,goals)
    assert row['status']=='failed' and row['diagnostic_tip_endpoint_h24_passed']
    assert row['actual_accepted_type'] is None


def test_all_unattempted_slots_still_render_complete_target_panel(tmp_path):
    pytest.importorskip('matplotlib')
    config=json.loads((ROOT/'configs/observed_two_row_pilot_v4.json').read_text())
    parent='two_row_reach_283200';slots=analysis.indexed_slots(parent,[],[])
    output=tmp_path/'all9.png'
    analysis.plot_target(dict(parent_id=parent,config=config),1,[s for s in slots if s['target']==1],{},output)
    assert output.read_bytes().startswith(b'\x89PNG') and output.stat().st_size>10000
