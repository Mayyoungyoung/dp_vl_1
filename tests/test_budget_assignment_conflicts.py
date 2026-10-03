import numpy as np
import pytest

from scripts import audit_budget_assignment_conflicts as a


def test_injective_and_surjective_exact_counts():
    assert len(a.assignments(16, 4)) == 43680
    assert len(a.assignments(2, 4)) == 14
    assert len(a.assignments(3, 4)) == 36
    assert a.assignments(1, 4).tolist() == [[0, 0, 0, 0]]
    for r in range(1, 5):
        for k in range(1, 5):
            assert all(len(set(row)) == min(r, k) for row in a.assignments(r, k))


def test_no_conflict_distinct_perfect_routes():
    r = a.analyze_cost(1-np.eye(4), np.arange(4), np.ones(4, bool), np.arange(4))
    assert not r['index_conflict']
    assert r['physical_consistency_at_every_stop']
    assert r['joint_chain_minimax_mean_mse_regret'] == 0


def test_ties_do_not_create_false_conflict():
    r = a.analyze_cost(np.zeros((4, 4)), np.arange(4), np.ones(4, bool), np.arange(4))
    assert r['free']['4']['tied_optima'] == 24
    assert not r['valid_slot_assigned_type_conflict']
    assert r['physical_consistency_at_every_stop']


def test_real_assignment_conflict_but_no_new_physical_route():
    cost = np.array([[0., 1., 9., 9.], [0., 5., 9., 9.], [9., 9., 0., 9.], [9., 9., 9., 0.]])
    r = a.analyze_cost(cost, np.arange(4), np.ones(4, bool), np.arange(4))
    assert r['index_conflict'] and r['valid_slot_assigned_type_conflict']
    assert not r['physical_consistency_at_every_stop']
    # Keeping [0,1] costs +2 at K2; keeping [1,0] costs +1 at K1.
    assert r['joint_chain_minimax_mean_mse_regret'] == 1.
    assert r['actual_predicted_modes'] == list(range(4))


def test_same_type_reference_swaps_are_not_type_conflicts():
    cost = np.array([[0., 1., 9., 9.], [0., 5., 9., 9.], [9., 9., 0., 9.], [9., 9., 9., 0.]])
    r = a.analyze_cost(cost, [0, 0, 2, 3], [True]*4, [0, 0, 2, 3])
    assert r['index_conflict'] and not r['assigned_type_conflict']


def test_invalid_prefix_relabelling_not_valid_type_conflict():
    cost = np.array([[0., 1., 9., 9.], [0., 5., 9., 9.], [9., 9., 0., 9.], [9., 9., 9., 0.]])
    r = a.analyze_cost(cost, np.arange(4), [False, False, True, True], [-1, -1, 2, 3])
    assert r['assigned_type_conflict'] and not r['valid_slot_assigned_type_conflict']
    assert r['invalid_candidates'] == r['unknown_candidates'] == 2


def test_saturation_allows_repeats_when_r_less_than_k():
    r = a.analyze_cost([[0, 1], [1, 0], [0, 1], [1, 0]], [0, 1], [True]*4, [0, 1, 0, 1])
    assert not r['index_conflict']
    assert r['actual_unique_valid_types'] == 2
    assert r['joint_chain_minimax_mean_mse_regret'] == 0


def test_enumeration_matches_existing_hungarian_saturation_formula():
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(99)
    for references in (1, 2, 3, 4, 6, 9, 16):
        cost = rng.uniform(size=(4, references))
        result = a.analyze_cost(cost, np.arange(references), [False]*4, [-1]*4)
        for k in a.STOPS:
            matrix = cost[:k]
            if references < k:
                nearest, base = matrix.argmin(1), matrix.min(1)
                rows, cols = linear_sum_assignment((matrix-base[:, None]).T)
                nearest[cols] = rows
                expected = matrix[np.arange(k), nearest].mean()
            else:
                rows, cols = linear_sum_assignment(matrix)
                expected = matrix[rows, cols].mean()
            assert result['free'][str(k)]['optimal_mean_interior_mse'] == pytest.approx(expected, abs=1e-15)


def test_shared_baseline_algebra_and_endpoints_excluded():
    rng = np.random.default_rng(2)
    p, r, base = rng.normal(size=(4, 24, 3)), rng.normal(size=(6, 24, 3)), rng.normal(size=(24, 3))
    expected = np.square((p-base)[:, None, 1:-1] - (r-base)[None, :, 1:-1]).mean((-1, -2))
    np.testing.assert_allclose(a.interior_cost(p, r), expected, atol=1e-14)
    before = a.interior_cost(p, r)
    p[:, [0, -1]] = 1000
    np.testing.assert_array_equal(a.interior_cost(p, r), before)


@pytest.mark.parametrize('cost', [np.zeros((8, 4)), np.full((4, 4), np.nan), -np.ones((4, 4)), np.zeros((4, 0))])
def test_bad_matrix_rejected(cost):
    with pytest.raises(ValueError):
        a.analyze_cost(cost, np.arange(4), [True]*4, np.arange(4))


def test_bounds_and_no_output_overwrite(tmp_path):
    with pytest.raises(ValueError):
        a.assignments(16, 8)
    with pytest.raises(ValueError, match='fresh'):
        a.run(tmp_path)


def test_identity_rejects_reserved_before_pool_shape():
    ids = np.asarray(['multigate_v1_TRAIN_%05d' % i for i in range(768)])
    data = dict(parent_ids=ids, scene_ids=np.asarray([p+'_c00' for p in ids]), splits=np.full(768, 'TEST_LOCKED'))
    with pytest.raises(ValueError, match='TRAIN only'):
        a.validate_identities(data, {}, [])


def test_gate_threshold_fixed_and_invalids_retained():
    row = a.analyze_cost(1-np.eye(4), np.arange(4), [False]*4, [-1]*4)
    result = a.summarize([row]*20)
    assert result['gate_decision'] == 'stop_no_nontrivial_type_conflict'
    assert result['invalid_candidates'] == 80 and result['unknown_candidates'] == 80
