import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import diagnose_two_row_endpoint_ik as diagnostic


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_endpoint_ik_v1.json').read_text())


def test_exactly_six_points_four_conditions_no_extra_candidates():
    plan=diagnostic.query_plan(config())
    assert len(plan)==24 and [r['query_id'] for r in plan]==list(range(24))
    assert len({tuple(r['xyz']) for r in plan})==6
    for xyz in {tuple(r['xyz']) for r in plan}:
        rows=[r for r in plan if tuple(r['xyz'])==xyz]
        assert {(r['orientation'],r['ignore_collisions']) for r in rows}=={
            ('original',False),('original',True),('world_vertical',False),('world_vertical',True)}


def test_registered_budget_cannot_silently_expand():
    changed=config();changed['trials']=256
    with pytest.raises(ValueError):diagnostic.query_plan(changed)


def test_vertical_from_actual_recorded_tip_preserves_heading():
    q=np.array([-.0001992768666241318,.9926828145980835,.000705028185620904,.120749332010746])
    out=diagnostic.vertical_quaternion(q)
    rot=diagnostic.rotation(out);heading=diagnostic.rotation(q)[:2,0]
    assert np.allclose(rot[:,2],[0,0,-1],atol=1e-12)
    assert np.allclose(rot[:2,0],heading/np.linalg.norm(heading),atol=1e-12)
    assert np.allclose(rot.T@rot,np.eye(3),atol=1e-12)
    assert np.linalg.det(rot)==pytest.approx(1)
    assert np.allclose(out,diagnostic.vertical_quaternion(-q))


def test_undefined_horizontal_heading_rejected():
    with pytest.raises(ValueError):diagnostic.vertical_quaternion([0,np.sqrt(.5),0,np.sqrt(.5)])
    with pytest.raises(ValueError):diagnostic.rotation([0,0,0,0])


def test_measured_attachment_axis_not_guessed():
    pose=[.0005548205226659775,.00020272471010684967,.8641334176063538,
          -.0001992768666241318,.9926828145980835,.000705028185620904,.120749332010746]
    attachment=[-.025775179266929626,.002044394612312317,.9709378480911255]
    result=diagnostic.measured_axis_evidence(pose,attachment)
    assert result['attachment_to_tip_dot_local_positive_z']>.99
    assert 13<result['original_local_z_tilt_from_world_down_degrees']<15
    with pytest.raises(ValueError):diagnostic.measured_axis_evidence(pose,pose[:3])


def test_bounded_search_counts_and_restores_binding():
    original=lambda *a,**k:[0]*7
    sim=SimpleNamespace(simGetConfigForTipPose=original)
    captured={}
    def solve(xyz,**kwargs):
        captured.update(kwargs)
        sim.simGetConfigForTipPose();sim.simGetConfigForTipPose()
        return np.zeros((1,7))
    arm=SimpleNamespace(solve_ik_via_sampling=solve);record={}
    diagnostic.bounded_ik(arm,sim,diagnostic.query_plan(config())[1],np.array([0,1,0,0]),config(),record)
    assert record['lowlevel_config_search_calls']==2 and sim.simGetConfigForTipPose is original
    assert captured['trials']==128 and captured['max_configs']==1 and captured['max_time_ms']==10
    assert captured['ignore_collisions'] is True


def test_bounded_search_restores_binding_on_failure():
    original=lambda *a,**k:[]
    sim=SimpleNamespace(simGetConfigForTipPose=original)
    def fail(*args,**kwargs):
        sim.simGetConfigForTipPose()
        raise RuntimeError('no configuration')
    record={}
    with pytest.raises(RuntimeError):
        diagnostic.bounded_ik(SimpleNamespace(solve_ik_via_sampling=fail),sim,
            diagnostic.query_plan(config())[0],np.array([0,1,0,0]),config(),record)
    assert sim.simGetConfigForTipPose is original and record['lowlevel_config_search_calls']==1
    assert record['ik_seconds']>=0


@pytest.mark.parametrize('world_difference,pixel_equal,passed',[(0,True,True),(1e-12,True,False),(0,False,False)])
def test_strict_common_restore_has_no_tolerance(monkeypatch,world_difference,pixel_equal,passed):
    obs=SimpleNamespace(front_rgb=np.zeros((1,1,3)))
    monkeypatch.setattr(diagnostic.legacy,'canonical_restore',lambda *a:None)
    monkeypatch.setattr(diagnostic,'audited_world',lambda *a:{})
    monkeypatch.setattr(diagnostic.legacy,'compare_restore',lambda *a:{'max_abs':world_difference,'global_inventory_equal':True,'rgb_max_difference':0})
    monkeypatch.setattr(diagnostic.pilot,'strict_observation_difference',lambda *a:{'front_rgb_equal':pixel_equal})
    evidence,_,_=diagnostic.strict_restore(SimpleNamespace(get_observation=lambda:obs),[],{}, {},obs,2)
    assert evidence['passed'] is passed and evidence['canonicalization_steps']==20
