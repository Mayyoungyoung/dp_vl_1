import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import collect_two_row_canonical_repair as repair

ROOT=Path(__file__).resolve().parents[1]


def config():return json.loads((ROOT/'configs/observed_two_row_canonical_repair_v6.json').read_text())


@pytest.mark.parametrize('key,value',[
    ('role','TRAIN'),('route_parent_seed',283101),('requested_route_proposals',54),
    ('setup_path_budget',1),('initialization_ik_budget',1),('fallback_attempts',1),
    ('selected_target_indices',[1]),('old_dynamic_state_reproduced',True)])
def test_registration_rejects_scope_expansion(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):repair.validate_registration(c)


def test_guard_forbids_init_and_restore_but_counts_route_calls_and_restores_api():
    class Arm:
        def get_path(self):return self.solve_ik()
        def solve_ik(self):return 'configuration'
    original=Arm.get_path;phase=dict(name='initialization');counts={}
    with repair.phase_guard(Arm,SimpleNamespace(),phase,counts):
        with pytest.raises(RuntimeError):Arm().get_path()
        phase['name']='restore'
        with pytest.raises(RuntimeError):Arm().solve_ik()
        phase['name']='route';assert Arm().get_path()=='configuration'
    assert Arm.get_path is original
    assert counts=={'initialization:get_path':1,'restore:solve_ik':1,'route:get_path':1,'route:solve_ik':1}


def test_source_hash_and_registered_joints_are_mandatory(tmp_path):
    c=config();c['source_v4']=str(tmp_path/'v4');c['source_v5']=str(tmp_path/'v5')
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);repair.pilot.write_json(path,value);return repair.batch.digest(path)
    anchor=Path(c['source_v4'])/c['anchor_file']
    c['anchor_sha256']=save(anchor,dict(state=dict(_robot=dict(arm_joints=c['canonical_arm_joints'],gripper_joints=c['canonical_gripper_joints']))))
    reg=json.loads((ROOT/'configs/observed_two_row_layout4_v5.json').read_text())
    base=json.loads((ROOT/'configs/observed_two_row_pilot_v4.json').read_text())
    plans=repair.batch.build_plan(reg,base)
    c['registration_sha256']=save(Path(c['source_v5'])/'registration.json',dict(parent_plan=plans))
    for plan in plans:
        seed=plan['config']['seed'];name='raw/%d/%s/before_preparation.json'%(seed,plan['parent_id'])
        c['v5_before_preparation_sha256'][str(seed)]=save(Path(c['source_v5'])/name,dict(state={}))
    actual,q,g=repair.load_sources(c)
    assert len(actual)==4 and q.shape==(7,) and g.shape==(2,)
    changed=copy.deepcopy(c);changed['canonical_arm_joints'][0]+=.1
    with pytest.raises(ValueError,match='joint values'):repair.load_sources(changed)
    anchor.write_text('{}')
    with pytest.raises(ValueError,match='SHA mismatch'):repair.load_sources(c)


def test_settling_collision_is_fatal_and_step_api_restored(monkeypatch):
    calls=[];scene=SimpleNamespace(step=lambda:calls.append('step'));task=SimpleNamespace(_scene=scene)
    original=scene.step;counts={}
    monkeypatch.setattr(repair.legacy,'robot_collision_check',lambda *args:'arm_environment')
    with pytest.raises(RuntimeError,match='fixed canonical settling'):
        with repair.monitor_settling(task,[],[],counts):scene.step()
    assert scene.step is original and calls==['step']
    assert counts=={'settling_steps':1,'settling_collision':'arm_environment'}


def test_nonzero_restore_difference_keeps_evidence_and_rejects(monkeypatch):
    task=SimpleNamespace(_scene=SimpleNamespace(step=lambda:None));check=dict(passed=False,world=dict(max_abs=1e-12))
    monkeypatch.setattr(repair.endpoint,'strict_restore',lambda *args:(check,{},None))
    with pytest.raises(RuntimeError) as error:
        repair.restore_check(task,[],dict(snapshot={},reference={},initial=None),[],[])
    assert error.value.restore_evidence['world']['max_abs']==1e-12


def test_route_thresholds_unchanged_and_wrong_endpoint_rejected():
    c=json.loads((ROOT/'configs/observed_two_row_pilot_v4.json').read_text())
    goal=np.asarray(c['goal_xyz']);sequence=('negative_y','negative_y')
    xyz=np.vstack([c['entry_xyz'],repair.pilot.proposed_waypoints(goal[1],sequence,c)])
    fields,_,passed=repair.route_acceptance(xyz,goal,1,c)
    assert passed and fields['actual_route_type']==sequence
    wrong=xyz.copy();wrong[-1,0]+=.031
    fields,_,passed=repair.route_acceptance(wrong,goal,1,c)
    assert not passed and fields['endpoint_error_m']>.03


@pytest.mark.parametrize('fatal,shutdown,expected',[(None,None,0),('runtime failure',None,1),(None,'shutdown failure',1)])
def test_exit_reports_runtime_errors_but_not_completed_negative_gates(tmp_path,monkeypatch,fatal,shutdown,expected):
    path=tmp_path/'config.json';path.write_text('{}')
    monkeypatch.setattr(repair,'run',lambda *args:dict(status='initialization_gate_closed',fatal_error=fatal,shutdown_error=shutdown))
    assert repair.main(['--config',str(path),'--output',str(tmp_path/'new')])==expected
