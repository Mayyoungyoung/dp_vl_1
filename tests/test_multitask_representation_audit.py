import numpy as np
from scripts.audit_multitask_train_representation import route_diagnostics


def test_floating_endpoint_is_not_forced_into_observed_surface():
    xyz=np.array([[0.,0.,1.],[0.,0.,1.1],[0.,0.,1.2]])
    poses=np.c_[xyz,np.tile([0.,0.,0.,1.],(3,1))];opened=np.array([1.,0.,0.])
    result=route_diagnostics(poses,opened,xyz,opened,np.array([[0.,0.,1.]]))
    assert result['event_sequence_preserved'] and result['phase_boundary_max_error_m']==0
    assert result['endpoint_nearest_observed_point_m']>.19
    assert not result['hard_surface_anchor_with_5cm_axis_residual_has_support']
    assert result['endpoint_error_m']==0


def test_missing_gripper_phase_is_reported_without_relabelling():
    xyz=np.array([[0.,0.,0.],[.1,0.,0.],[.1,.1,0.]])
    poses=np.c_[xyz,np.tile([0.,0.,0.,1.],(3,1))];opened=np.array([1.,0.,1.])
    before=opened.copy()
    result=route_diagnostics(poses,opened,xyz,np.ones(3),xyz)
    assert not result['event_sequence_preserved'] and result['phase_boundary_max_error_m'] is None
    assert np.array_equal(opened,before)
    assert result['raw_sample_to_resampled_polyline_max_m']==0
