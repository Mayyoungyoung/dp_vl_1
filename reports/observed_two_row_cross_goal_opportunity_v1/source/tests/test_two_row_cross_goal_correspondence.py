import json
import numpy as np
import pytest

from scripts.audit_two_row_cross_goal_correspondence import (
    IDS, PARENTS, arc_sample, balanced_pair, describe, digest, distance,
    load_pool, prediction_screen, selected_lines, summarize_pairs, train_path)


def curve(y=0., z=.5):
    return np.array([[-1.,y,z],[0.,y,z],[1.,y,z],[2.,y,z]])


def item(mode, y=0., z=.5):
    return dict(type=mode, descriptor=describe(curve(y,z), [0.,1.], 1.))


def test_vertex_crossing_count_direction_and_height_are_physical_not_h24_indices():
    a=describe(curve(.2,.7), [0.,1.], 1.)
    b=describe(np.array([[-1,.2,.7],[-.9,.2,.7],[.3,.2,.7],[2,.2,.7]]), [0.,1.], 1.)
    assert a['simple_forward'] and a['crossing_counts']==[1,1]
    np.testing.assert_allclose(a['crossing_yz'],[[.2,-.3],[.2,-.3]])
    assert distance(a,b)['crossing_yz_m']==pytest.approx(0.)
    np.testing.assert_allclose(arc_sample(curve(.2,.7)),arc_sample(np.array([[-1,.2,.7],[2,.2,.7]])))


@pytest.mark.parametrize('path,reason',[
    ([[-1,0,.5],[.5,0,.5],[-.5,0,.5],[2,0,.5]],'multiple_row_crossings'),
    ([[-1,0,.5],[0,0,.5],[-1,0,.5]],'tangent_or_endpoint_touch'),
    ([[-1,0,.5],[0,0,.5],[0,1,.5],[2,1,.5]],'coplanar_segment'),
    ([[-1,0,.5],[-.5,0,.5]],'missing_row_crossing'),
    ([[2,0,.5],[-1,0,.5]],'backward_crossing')])
def test_ambiguous_routes_remain_explicit_not_made_into_correspondence(path,reason):
    result=describe(path,[0.,1.],1.)
    assert not result['simple_forward'] and reason in result['ambiguity']
    assert distance(result,describe(curve(),[0.,1.],1.)) is None


def test_same_type_does_not_imply_close_geometry_and_unknown_is_not_negative():
    mode=['middle','middle']
    left=[item(mode),item(None,10.)]
    right=[item(mode,3.),item(['over','middle'],.1),item(None,-10.)]
    out=balanced_pair(left,right)
    assert out['same']['crossing_yz_m']==pytest.approx(3.)
    assert out['different']['crossing_yz_m']==pytest.approx(.1)
    assert out['all_route_pairs']==6 and out['known_route_pairs']==2
    assert out['unknown_routes_left']==out['unknown_routes_right']==1


def test_type_pairs_are_balanced_despite_repeated_examples():
    a=['middle','middle']; b=['over','middle']
    left=[item(a),item(b,1.)]; right=[item(a,.1),item(b,1.8)]
    original=balanced_pair(left,right)
    repeated=balanced_pair([left[0]]*40+[left[1]],right)
    assert original['same']==repeated['same']
    assert original['different']==repeated['different']
    assert original['known_route_pairs']!=repeated['known_route_pairs']


def test_chord_subtraction_is_description_only_and_long_arc_not_removed():
    a=np.array([[-1.,0,.5],[.1,0,4.],[2,0,.5]])
    original=a.copy();d=describe(a,[0.,1.],1.)
    assert d['long_arc'] and d['simple_forward']
    assert d['maximum_z_m']==4. and d['length_m']>6.
    np.testing.assert_array_equal(a,original)
    assert np.linalg.norm(d['chord_residual'])>0


def test_selection_never_decodes_dev_payload_and_rejects_missing_or_role_changed(tmp_path):
    path=tmp_path/'supervision.jsonl'
    rows=[dict(id=i,parent_id=i.rsplit('_target',1)[0],split='TRAIN') for i in sorted(IDS)]
    forbidden='{"id": "two_row_reach_283264_target0", THIS PAYLOAD MUST NOT BE DECODED\n'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows)+forbidden)
    assert set(selected_lines(path))==IDS
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1])+forbidden)
    with pytest.raises(ValueError,match='All original48'): selected_lines(path)
    rows[0]['split']='DEV_MODEL';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='non-TRAIN'): selected_lines(path)


def test_raw_path_forbids_other_parents_roles_and_escape(tmp_path):
    parent=PARENTS[0];base=tmp_path/'parents'/'TRAIN'/parent;base.mkdir(parents=True)
    assert train_path(base/'route.npz',tmp_path,parent)==base/'route.npz'
    for bad in [tmp_path/'parents'/'DEV_MODEL'/parent/'route.npz',base/'..'/'elsewhere.npz']:
        with pytest.raises(ValueError,match='outside selected TRAIN'):train_path(bad,tmp_path,parent)
    with pytest.raises(ValueError,match='Unregistered'):train_path(base/'route.npz',tmp_path,'two_row_reach_283216')


def test_pool_requires_all48_parent_ids_shapes_and_actual_hash(tmp_path):
    path=tmp_path/'pool.npz';ids=np.array(sorted(IDS));parents=np.array([i.rsplit('_target',1)[0] for i in ids])
    paths=np.zeros((48,4,24,3));events=np.ones((48,4,24))
    paths[0,0,0]=np.nan  # invalid predictions retained for downstream checker
    np.savez(path,scene_ids=ids,parent_ids=parents,paths=paths,gripper_open=events)
    assert len(load_pool(path,digest(path)))==48
    with pytest.raises(ValueError,match='SHA'):load_pool(path,'0'*64)
    np.savez(path,scene_ids=ids,parent_ids=parents,paths=paths[:-1],gripper_open=events)
    with pytest.raises(ValueError,match='Exact48'):load_pool(path,digest(path))


def test_parent_summary_and_opportunity_gate_preserve_unevaluable_denominator():
    thresholds=dict(minimum_eligible_target_pairs=24,minimum_eligible_parents=12,
        minimum_two_common_type_pairs=12,maximum_median_same_different_ratio=.75,minimum_fraction_same_smaller=2/3)
    rows=[dict(parent_id='p',same=None,different=None,known_common_count=0)]*48
    out=summarize_pairs(rows,thresholds)
    assert out['actual_target_pairs']==48 and out['eligible_pairs']==0
    assert out['parent_equal_mean_same'] is None and not out['reference_structure_screen_passed']
    metrics=lambda v:dict(crossing_yz_m=v,lateral_m=v,height_m=v,chord_residual_m=v)
    rows=[dict(parent_id='a',same=metrics(1.),different=metrics(2.),known_common_count=2)]*3
    rows+=[dict(parent_id='b',same=metrics(3.),different=metrics(4.),known_common_count=2)]
    out=summarize_pairs(rows,thresholds)
    assert out['pair_equal_mean_same']['crossing_yz_m']==1.5
    assert out['parent_equal_mean_same']['crossing_yz_m']==2.


def test_reference_separation_alone_cannot_pass_prediction_gap_or_unknown_denominator():
    thresholds=dict(minimum_colliding_candidates=4,minimum_clear_candidates=8,minimum_parents_per_group=4,
        minimum_collision_excess_m=.02,minimum_collision_minus_clear_m=.02,minimum_collision_fraction_excess=.5)
    rows=[dict(id=PARENTS[0]+'_target0',semantic_correct=True,collision=False,known_type=False,
        simple_forward=False,mean_excess_m=None)]*192
    out=prediction_screen(rows,thresholds)
    assert out['all_candidate_slots']==192 and out['unknown_type_slots']==192
    assert out['eligible_semantic_correct_slots']==0 and not out['prediction_gap_screen_passed']


def test_prediction_gap_requires_goal_correct_collision_association_across_parents():
    thresholds=dict(minimum_colliding_candidates=4,minimum_clear_candidates=8,minimum_parents_per_group=4,
        minimum_collision_excess_m=.02,minimum_collision_minus_clear_m=.02,minimum_collision_fraction_excess=.5)
    rows=[]
    for parent in PARENTS[:4]:
        rows.append(dict(id=parent+'_target0',semantic_correct=True,collision=True,known_type=True,
            simple_forward=True,mean_excess_m=.04))
        rows.extend([dict(id=parent+'_target1',semantic_correct=True,collision=False,known_type=True,
            simple_forward=True,mean_excess_m=.005)]*2)
    assert prediction_screen(rows,thresholds)['prediction_gap_screen_passed']
    # A wrong-goal outlier must not provide evidence about target-correct drift.
    for row in rows:
        if row['collision']: row['semantic_correct']=False
    assert not prediction_screen(rows,thresholds)['prediction_gap_screen_passed']
