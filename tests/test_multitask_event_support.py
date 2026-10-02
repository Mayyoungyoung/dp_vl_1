import copy
import numpy as np
import pytest
from scripts.audit_multitask_event_support import event_locations,support,selected_train
from scripts.observation_collect_multitask import registration


def test_first_closing_event_keeps_before_after_without_final_endpoint_substitution():
    path=np.array([[0,0,0],[.1,0,0],[.11,0,0],[1.,0,0]])
    result=event_locations(path,np.array([1,1,0,0]),np.array([[.1,0,0]]))
    assert result['primary_landmark']=='first_close_after' and result['first_close_index']==2
    assert result['first_close_before']['nearest_observed_l2_m']==0
    assert result['first_close_after']['nearest_observed_l2_m']==pytest.approx(.01)
    assert result['endpoint']['nearest_observed_l2_m']==pytest.approx(.9)


def test_no_transition_reports_endpoint_and_open_only_does_not_invent_grasp():
    path=np.zeros((3,3));points=np.ones((1,3))
    result=event_locations(path,np.ones(3),points)
    assert result['category']=='constant_open' and result['primary_landmark']=='endpoint'
    result=event_locations(path,np.array([0,0,1]),points)
    assert result['primary_landmark'] is None and result['first_close_after'] is None


def test_support_is_unlabelled_radius_count_and_rejects_nonfinite():
    result=support([0,0,0],np.array([[.03,.04,0],[.10,0,0]]))
    assert result['nearest_observed_l2_m']==pytest.approx(.05)
    assert result['observed_point_counts']['0.05']==1
    assert result['observed_point_counts']['0.1']==2
    with pytest.raises(ValueError):support([np.nan,0,0],np.zeros((1,3)))


def test_train_selection_excludes_every_other_role_and_refuses_changed_parent():
    plan=registration(282000,'validated-five-v1','interleaved-early-dev-v1')
    manifest=dict(protocol='multitask_closed_prefix_snapshot_v1',selected_requested_parents=[r for r in plan['parents'] if (r['split']=='TRAIN' and r['parent_index']<8) or (r['split']=='DEV_MODEL' and r['parent_index'] in (16,17))])
    assert len(selected_train(manifest))==48
    changed=copy.deepcopy(manifest);changed['selected_requested_parents'][0]['split']='DEV_MODEL'
    with pytest.raises(ValueError,match='forty-eight TRAIN'):selected_train(changed)
