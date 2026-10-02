import numpy as np
from scripts.audit_two_row_train_reference_quality import endpoint_support,trajectory_statistics


def test_axis_support_is_not_confused_with_euclidean_bound_or_invalid_pixels():
    xyz=np.array([[.049,.049,.049],[0.,0.,0.]])
    result=endpoint_support(np.zeros(3),xyz,[True,False])
    assert result['axis_bound_representable'] and result['nearest_L2_m']>.05
    assert not endpoint_support(np.zeros(3),[[.051,0,0]],[True])['axis_bound_representable']
    assert not endpoint_support(np.zeros(3),xyz,[False,False])['axis_bound_representable']


def test_long_high_arc_is_described_not_shortened_or_filtered():
    path=np.array([[0,0,.865],[.2,0,1.8],[.4,0,.84]])
    original=path.copy();result=trajectory_statistics(path)
    assert result['maximum_world_z_m']==1.8 and result['length_m']>1.8
    assert np.array_equal(path,original)
