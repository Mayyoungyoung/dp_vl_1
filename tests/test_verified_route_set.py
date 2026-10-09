import numpy as np
from routeset.verified_route_set import (signed_clearances, certified_radii,
                                        verify_reach, build_targets, build_set_targets)
from routeset.geometry import segment_aabb_intersection


def test_segment_interior_collision_not_endpoint_probe():
    p = np.array([[[-1., 0, .5], [1., 0, .5]]])
    assert signed_clearances(p, np.array([[0., 0, .5]]), np.ones((1, 3))*.1)[0, 0] < 0


def test_region_random_extreme_corners_protect_entire_segments():
    p = np.array([[[-1., .3, .5], [0, .3, .5], [1., .3, .5]]])
    centers, halves = np.array([[0., 0, .5]]), np.ones((1, 3))*.1
    radius = certified_radii(p, centers, halves, .1, cap=.5)
    assert (radius[:, [0, -1]] == 0).all()
    rng = np.random.default_rng(3)
    variants = p + rng.choice([-1, 1], (100, 3, 3))*radius[..., None]
    assert not segment_aabb_intersection(variants[:, :-1], variants[:, 1:], centers-.12, centers+.12).any()


def toy():
    # Continuous x endpoints; middle y distinguishes route classes.
    refs = np.array([[[0., 0, 1], [.5, -1, 1], [1., 0, 1]],
                     [[0., 0, 1], [.5, 1, 1], [1., 0, 1]]])
    events = np.ones((2, 3))
    def check(paths, ev):
        words = ['left' if x[1, 1] < 0 else 'right' for x in paths]
        return np.ones(len(paths), dtype=bool), words
    return refs, events, check


def test_correct_unreferenced_route_preserved_duplicate_reallocated():
    refs, ev, check = toy()
    p = refs[[0, 0]].copy(); p[0, 1, 1] = -1.4
    target, _, audit = build_targets(p, ev, refs, ev, ['left', 'right'], np.zeros((2, 3)),
        [True, True], ['left', 'left'], 'project', np.random.default_rng(2), check)
    np.testing.assert_array_equal(target[0], p[0])
    np.testing.assert_array_equal(target[1], refs[1])
    assert audit['protected'].tolist() == [True, False]
    assert audit['target_words'] == ['left', 'right']


def test_gate_is_simple_control_not_reassignment():
    refs, ev, check = toy()
    p = refs[[0, 0]].copy()
    t, _, a = build_targets(p, ev, refs, ev, ['left', 'right'], np.zeros((2, 3)),
        [True, True], ['left', 'left'], 'gate', np.random.default_rng(4), check)
    np.testing.assert_array_equal(t[0], p[0])
    assert a['protected'].sum() == 1


def test_mode_changing_projection_falls_back_to_valid_witness():
    refs, ev, check = toy()
    p = refs[[0, 0]].copy()
    t, _, a = build_targets(p, ev, refs, ev, ['left', 'right'], np.ones((2, 3))*3,
        [False, False], [None, None], 'project', np.random.default_rng(0), check)
    assert a['fallback'].sum() == 1
    assert set(check(t, ev)[1]) == {'left', 'right'}


def test_oversubscribed_does_not_overwrite_protected_new_mode():
    refs, ev, check = toy()
    p = refs[:1].copy()
    def novel_check(p, e): return np.ones(len(p), bool), ['novel']*len(p)
    t, _, a = build_targets(p, ev[:1], refs, ev, ['left', 'right'], np.zeros((2, 3)),
        [True], ['novel'], 'project', np.random.default_rng(0), novel_check)
    np.testing.assert_array_equal(t, p)
    assert a['oversubscribed'] and a['unmatched_modes'] == 2


def test_unknown_valid_signature_not_falsely_counted_as_distinct():
    refs, ev, check = toy()
    _, _, a = build_targets(refs, ev, refs, ev, ['left', 'right'], np.zeros((2, 3)),
        [True, True], [None, None], 'project', np.random.default_rng(0), check)
    assert not a['protected'].any()


def test_target_does_not_mutate_predictions_or_references_and_rng_matches():
    refs, ev, check = toy(); original = refs.copy(); states = []
    for arm in ['ordinary', 'gate', 'project']:
        rng = np.random.default_rng(11)
        build_targets(refs, ev, refs, ev, ['left', 'right'], np.zeros((2, 3)),
            [True, True], ['left', 'right'], arm, rng, check)
        states.append(rng.bit_generator.state)
    assert states[0] == states[1] == states[2]
    np.testing.assert_array_equal(refs, original)


def test_reach_checks_semantics_start_events_floor_and_continuous_geometry():
    current = dict(gripper_pose=np.array([-1., .3, .5]), gripper_open=np.array(1.))
    p = np.array([[[-1., .3, .5], [0., .3, .5], [1., .3, .5]]]*6)
    e = np.ones((6, 3)); goals = np.array([[1., .3, .5], [1., -.3, .5]])
    p[1, -1] = goals[1]; p[2, 0, 0] += .01; e[3, 1] = 0
    p[4, 1] = [0, 0, .5]; p[5, 1, 2] = -.1
    v = verify_reach(p, e, current, goals, 0, np.array([[0., 0, .5]]), np.ones((1, 3))*.1, .1)
    assert v.tolist() == [True, False, False, False, False, False]


def test_joint_assignment_avoids_arbitrary_first_slot_lock_counterexample():
    refs,ev,check=toy();p=refs[[0,0]].copy();p[0,1,1]=-.1
    hard,_,_=build_targets(p,ev,refs,ev,['left','right'],np.zeros((2,3)),
        [True,True],['left','left'],'project',np.random.default_rng(0),check)
    joint,_,audit=build_set_targets(p,ev,refs,ev,['left','right'],np.zeros((2,3)),
        [True,True],['left','left'],'set_point',np.random.default_rng(0),check)
    assert np.square(joint-p).sum()<np.square(hard-p).sum()
    assert set(audit['target_words'])=={'left','right'}
    np.testing.assert_array_equal(joint[1],p[1])


def test_joint_target_is_slot_permutation_equivariant_in_unique_optimum():
    refs,ev,check=toy();p=refs[[0,0]].copy();p[0,1,1]=-.1
    def solve(p):
        return build_set_targets(p,ev,refs,ev,['left','right'],np.zeros((2,3)),
            [True,True],['left','left'],'set_point',np.random.default_rng(0),check)[0]
    np.testing.assert_array_equal(solve(p)[::-1],solve(p[::-1]))


def test_joint_valid_distinct_unreferenced_set_has_zero_correction():
    refs,ev,check=toy();p=refs.copy();p[:,1,1]=[-1.4,1.4]
    for arm in ('set_point','set_project'):
        target,_,_=build_set_targets(p,ev,refs,ev,['left','right'],np.zeros((2,3)),
            [True,True],['left','right'],arm,np.random.default_rng(0),check)
        np.testing.assert_array_equal(target,p)


def test_joint_overbudget_keeps_valid_novel_mode_without_inventing_full_cover():
    refs,ev,_=toy();p=refs[:1].copy();p[0,1,1]=0
    def check(p,e):
        return np.ones(len(p),bool),['novel' if x[1,1]==0 else 'left' if x[1,1]<0 else 'right' for x in p]
    target,_,audit=build_set_targets(p,ev[:1],refs,ev,['left','right'],np.zeros((2,3)),
        [True],['novel'],'set_point',np.random.default_rng(0),check)
    np.testing.assert_array_equal(target,p)
    assert audit['oversubscribed'] and audit['unmatched_modes']==2


def test_cross_mode_zero_distance_box_is_not_a_valid_zero_cost_edge():
    refs,ev,check=toy();p=refs.copy();p[:,1,1]=[-1.4,1.4]
    target,_,_=build_set_targets(p,ev,refs,ev,['left','right'],np.ones((2,3))*5,
        [True,True],['left','right'],'set_project',np.random.default_rng(0),check)
    np.testing.assert_array_equal(target,p)
