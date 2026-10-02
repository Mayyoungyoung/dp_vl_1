import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.collect_observed_two_row_pilot import (PASSAGES, crossing_signature, geometry,
    proposed_waypoints, strict_observation_difference, validate_config)
from scripts import collect_observed_two_row_pilot as collector


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v1.json"
    return json.loads(path.read_text())


def test_nine_proposals_are_geometry_only_not_simulation_success(config):
    centers, halves = validate_config(config)
    actual = set()
    for goal in config["goal_xyz"]:
        for sequence in itertools.product(PASSAGES, repeat=2):
            xyz = np.vstack([config["entry_xyz"], proposed_waypoints(goal, sequence, config)])
            assert collector.legacy.tip_polyline_clear(xyz, centers, halves, .02)
            assert crossing_signature(xyz, config) == sequence
            sampled = collector.legacy.resample(xyz, 24)
            assert collector.legacy.tip_polyline_clear(sampled, centers, halves, .02)
            assert crossing_signature(sampled, config) == sequence
            actual.add(sequence)
    assert len(actual) == 9
    assert not config["reference_set_complete"] and config["all_solution_count"] is None


def test_actual_crossings_cannot_be_replaced_with_guide_ids(config):
    xyz = np.vstack([config["entry_xyz"], proposed_waypoints(config["goal_xyz"][1], ("middle", "middle"), config)])
    assert crossing_signature(xyz, config) == ("middle", "middle")
    assert crossing_signature(xyz[::-1], config) is None
    assert crossing_signature(xyz[:3], config) is None
    xyz[2:5, 1] = -.25
    assert crossing_signature(xyz, config) != ("middle", "middle")


def test_collision_boundary_and_overpass_are_not_new_lateral_openings(config):
    top = config["post_base_z"]+config["post_size_xyz"][2]
    over = np.array([[0., 0., top+.1], [.5, 0., top+.1]])
    assert crossing_signature(over, config) == ("over", "over")
    inside = np.array([[0., -.11, .865], [.5, -.11, .865]])
    assert crossing_signature(inside, config) is None
    assert not collector.legacy.tip_polyline_clear(inside, *geometry(config), clearance=.02)
    boundary = np.array([[0., 0., top], [.5, 0., top]])
    assert crossing_signature(boundary, config) is None


def test_backtracking_across_rows_and_nonfinite_are_unknown(config):
    xyz = np.array([[0.,0.,.865],[.4,0.,.865],[.05,0.,.865],[.5,0.,.865]])
    assert crossing_signature(xyz, config) is None
    xyz[1, 2] = np.nan
    assert crossing_signature(xyz, config) is None


def test_strict_observation_audit_includes_depth_camera_and_state():
    from types import SimpleNamespace
    observed = SimpleNamespace(front_rgb=np.zeros((2,2,3), dtype=np.uint8), front_depth=np.ones((2,2)),
        gripper_pose=np.zeros(7), gripper_open=1., misc={"front_camera_intrinsics":np.eye(3),"front_camera_extrinsics":np.eye(4)})
    changed = copy.deepcopy(observed)
    assert all(strict_observation_difference(observed, changed).values())
    changed.front_depth[0,0] += 1e-7
    changed.misc["front_camera_extrinsics"][0,3] += 1e-7
    differences = strict_observation_difference(observed, changed)
    assert not differences["front_depth_equal"] and not differences["front_camera_extrinsics_equal"]
    assert differences["front_rgb_equal"]


def test_proposal_budget_and_role_cannot_silently_change(config):
    config["requested_route_proposals"] = 100
    with pytest.raises(ValueError):
        validate_config(config)


def test_v2_explicit_plane_guides_add_calls_without_changing_physical_or_type_rules(config):
    path=Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v2.json"
    changed=json.loads(path.read_text())
    centers,halves=validate_config(changed)
    assert all(changed[k]==value for k,value in config.items() if k!="protocol")
    assert np.array_equal(geometry(config)[0],centers)
    for goal in changed["goal_xyz"]:
        for sequence in itertools.product(PASSAGES,repeat=2):
            original=proposed_waypoints(goal,sequence,config)
            waypoints=proposed_waypoints(goal,sequence,changed)
            assert len(original)==7 and len(waypoints)==9
            assert np.array_equal(waypoints[[0,2,3,4,5,7,8]],original)
            xyz=np.vstack([changed["entry_xyz"],waypoints])
            assert crossing_signature(xyz,changed)==sequence
            assert collector.legacy.tip_polyline_clear(xyz,centers,halves,.02)
            sampled=collector.legacy.resample(xyz,24)
            assert crossing_signature(sampled,changed)==sequence
            assert collector.legacy.tip_polyline_clear(sampled,centers,halves,.02)


@pytest.mark.parametrize("failed_stage", ["planning", "simulation"])
def test_failed_segment_records_actual_cost_without_filling_later_calls(config, monkeypatch, failed_stage):
    from types import SimpleNamespace
    def get_path(*args, **kwargs):
        if failed_stage == "planning":
            raise RuntimeError("fixture planning failure")
        return SimpleNamespace(step=lambda: True)
    task = SimpleNamespace(_robot=SimpleNamespace(arm=SimpleNamespace(get_path=get_path)),
                           _scene=SimpleNamespace(step=lambda: None))
    monkeypatch.setattr(collector, "state_sample", lambda task: (np.zeros(7), 1., np.zeros(7), np.zeros(2)))
    monkeypatch.setattr(collector.legacy, "robot_collision_check", lambda *args: "fixture collision")
    record, trace = {"simulated_steps":0}, []
    with pytest.raises(RuntimeError):
        collector.execute(task, [[0.,0.,0.],[1.,0.,0.]], [0.,0.,0.,1.], [], [], trace, record, config)
    assert len(record["planning_segments"]) == 1
    segment = record["planning_segments"][0]
    assert segment[failed_stage+"_status"] == "failed"
    assert segment["planning_seconds"] >= 0 and segment["simulation_seconds"] >= 0
    assert segment["simulated_steps"] == len(trace) == int(failed_stage == "simulation")
    totals = {}
    collector.add_execution_totals(totals, record)
    assert totals["get_path_calls"] == 1


def v4_config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v4.json").read_text())


def test_v4_changes_only_post_height_and_preregistered_scope():
    old=json.loads((Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v2.json").read_text())
    new=v4_config();collector.validate_config(new)
    for key,value in old.items():
        if key not in {'protocol','seed','post_size_xyz','requested_route_proposals'}:assert new[key]==value
    assert new['post_size_xyz']==[.035,.035,.14] and old['post_size_xyz']==[.035,.035,.16]
    assert new['seed']!=old['seed'] and collector.registered_target_indices(new)==[1]
    assert collector.registered_target_indices(old)==[0,1,2]
    assert len(new['goal_xyz'])==3 and new['requested_route_proposals']==9


def test_v4_nine_central_paths_keep_original_guides_and_raw_h24_rules():
    new=v4_config();centers,halves=collector.validate_config(new)
    old=json.loads((Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v2.json").read_text())
    for sequence in itertools.product(PASSAGES,repeat=2):
        guides=proposed_waypoints(new['goal_xyz'][1],sequence,new)
        assert len(guides)==9 and np.array_equal(guides,proposed_waypoints(old['goal_xyz'][1],sequence,old))
        raw=np.vstack([new['entry_xyz'],guides])
        for xyz in (raw,collector.legacy.resample(raw,24)):
            assert crossing_signature(xyz,new)==sequence
            assert collector.legacy.tip_polyline_clear(xyz,centers,halves,.02)
    top=new['post_base_z']+new['post_size_xyz'][2]
    assert crossing_signature([[0,0,top+.1],[.5,0,top+.1]],new)==('over','over')
    assert crossing_signature([[0,0,top],[.5,0,top]],new) is None


def test_v4_unknown_duplicate_and_over_do_not_rewrite_legacy_threshold(config):
    references=[('negative_y','middle'),('middle','middle'),('positive_y','middle'),
        ('over','middle'),('over','over'),('over','over'),None]
    current=collector.summarize_reference_types('fixture',references,v4_config())
    legacy=collector.summarize_reference_types('fixture',references,config)
    assert current['valid_references']==7 and current['distinct_classified']==5
    assert current['unknown_valid_references']==1 and current['distinct_lateral_sequences']==3
    assert current['preregistered_feasibility_met'] and not current['has_more_than_four_valid_lateral_sequences']
    assert not legacy['preregistered_feasibility_met'] and legacy['feasibility_type_policy']=='lateral_only'


@pytest.mark.parametrize('field,value',[('selected_target_indices',[0]),('requested_route_proposals',27),
    ('feasibility_type_policy','lateral_only'),('minimum_distinct_valid_types',4)])
def test_v4_rejects_unregistered_target_budget_or_primary_type_policy(field,value):
    new=v4_config();new[field]=value
    with pytest.raises(ValueError):collector.validate_config(new)
