"""Pure NumPy interface tests; actual processor validation is a separate CPU job."""
import numpy as np
import pytest

from routeset.vlm_route_serialization import depth_image
from scripts.audit_vlm_depth_interface import (PARENTS, PROTOCOL, compare_arrays,
                                             select_samples, unpack_pixels, unnormalize_pixels)


def test_patch_permutation_and_normalization_inverse_exact_then_byte_tolerance():
    p, m, tp = 2, 2, 2
    raw = np.arange(8*12*3, dtype=np.float32).reshape(8, 12, 3) % 256
    mean, std, factor = [.48, .46, .41], [.27, .26, .28], 1/255
    normalized = (raw*factor-mean)/std
    original = np.repeat(normalized.transpose(2, 0, 1)[None], tp, axis=0)
    before = original.reshape(1, tp, 3, 2, m, p, 3, m, p)
    flat = before.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(24, 3*tp*p*p)
    actual = unpack_pixels(flat, [1, 4, 6], p, m, tp)
    np.testing.assert_array_equal(actual, normalized)
    np.testing.assert_allclose(unnormalize_pixels(actual, mean, std, factor), raw, atol=.00003)
    with pytest.raises(ValueError, match='patch shape'):
        unpack_pixels(flat, [2, 4, 6], p, m, tp)
    broken = flat.copy(); broken[0, p*p] += 1
    with pytest.raises(ValueError, match='temporal'):
        unpack_pixels(broken, [1, 4, 6], p, m, tp)


def test_identity_depth_quantization_keeps_mm_and_unknown_validity():
    depth = np.array([[.2551, .2562], [np.nan, .9019]])
    encoded = depth_image(depth); valid = np.isfinite(depth)
    metric = np.nan_to_num(depth)
    result, decoded, after, support = compare_arrays(depth, encoded, encoded.astype(float), metric, metric, valid, valid)
    assert result['raw_quantization_abs_m']['max'] <= .0005
    assert result['resized_B_intermediate_pixels'] == 0
    assert result['validity_disagreement_with_nearest'] == 0
    assert result['resized_B0_pixels'] == 1 and support.sum() == 3
    np.testing.assert_array_equal(after, encoded)
    assert decoded[0, 1] == .256


def test_mock_byte_carry_interpolation_error_and_invalid_blue_are_separate():
    depth = np.array([[.255, .256]])
    # An explicitly synthetic channel-rounding example, not a runtime resize claim.
    after = np.array([[[1., 128., 255.], [0., 128., 128.]]])
    metric = np.array([[.2555, .128]])
    result, _, _, supported = compare_arrays(depth, depth_image(depth), after, metric,
        np.array([[.256, 0.]]), np.array([[True, False]]), np.array([[1., .5]]))
    assert result['byte_decode_error_gt_10cm'] == 1
    assert abs(result['byte_decode_vs_float_metric_bicubic_abs_m']['max']-.1285) < 1e-12
    assert result['resized_B_intermediate_pixels'] == 1
    assert supported.sum() == 1


def test_exact_eight_train_first_observation_selection_rejects_role_or_count_changes():
    plan = dict(protocol=PROTOCOL, samples=[dict(parent_id=p, id=p+'_target0') for p in PARENTS])
    rows = [dict(parent_id=p, id=p+'_target%d'%j, split='TRAIN', instruction='reach', image='unopened.png')
            for p in PARENTS for j in (2, 1, 0)]
    rows.append(dict(parent_id='obstacle_reach_272120', id='unopened_locked', split='TEST_LOCKED'))
    assert [r['id'] for r in select_samples(rows, plan)] == [r['id'] for r in plan['samples']]
    changed = [dict(r) for r in rows]; changed[0]['split'] = 'DEV_MODEL'
    with pytest.raises(ValueError, match='role'):
        select_samples(changed, plan)
    with pytest.raises(ValueError, match='eight'):
        select_samples(rows, dict(plan, samples=plan['samples'][:-1]))
    with pytest.raises(ValueError, match='missing'):
        select_samples([r for r in rows if r['id'] != PARENTS[0]+'_target0'], plan)
