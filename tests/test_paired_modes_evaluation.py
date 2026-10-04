import numpy as np
import pytest
pytest.importorskip('torch')
from scripts.paired_modes_data import registration
from scripts.observed_layout_variation import geometry
from scripts.evaluate_paired_modes import check_candidates
from scripts.paired_modes_reliability import reliability_metrics


def test_full_segment_collision_remains_invalid_with_correct_endpoints():
    plan=registration()['parent_plan'][1];cfg=plan['config'];cs,hs=geometry(cfg)
    start=np.asarray(cfg['entry_xyz']);goal=np.asarray(cfg['goal_xyz'][1]);x=cfg['row_x'][0]
    # Deliberately place a segment through the certified closed center.
    y=np.mean(cfg['post_y'][0]);z=cfg['post_base_z']+.06
    path=np.array([start,[x-.08,y,z],[x+.08,y,z],goal])
    label=dict(semantic_targets=dict(centers=cfg['goal_xyz'],target_index=1,tolerance=.03),route_types=[])
    current=dict(gripper_pose=start,gripper_open=np.array(1.));truth=dict(obstacle_centers=cs,obstacle_halfsizes=hs)
    _,c=check_candidates(path[None],np.ones((1,4)),label,current,truth,cfg)
    assert c[0]['semantic_goal_correct'] and not c[0]['tip_segments_clear'] and not c[0]['TipValid']


def test_reliability_counts_confident_failures_and_family_clustering():
    logits=np.array([[3.,1.],[-3.,-1.],[2.,-2.]])
    data=dict(labels=np.array([[0,1],[0,0],[1,0]]),paths=np.zeros((3,2,4,3)),
              parents=np.array(['paired_family_1_open','paired_family_1_closed','paired_family_1_shifted']))
    m=reliability_metrics(logits,data)
    assert m['parent_count']==1 and m['selected_valid']==1/3
    assert m['confident_08']['candidates']==2 and m['confident_08']['observed_validity']==.5
    assert np.isfinite(m['nll']) and m['selected_reliability_bins_ece']>0


def test_route_under_workspace_is_not_valid_diversity():
    cfg=registration()['parent_plan'][0]['config'];cs,hs=geometry(cfg)
    start=np.asarray(cfg['entry_xyz']);goal=np.asarray(cfg['goal_xyz'][1])
    path=np.array([start,[0,-.4,.7],[.52,-.4,.7],[.52,goal[1],.84],goal])
    label=dict(semantic_targets=dict(centers=cfg['goal_xyz'],target_index=1,tolerance=.03),route_types=[])
    metrics,c=check_candidates(path[None],np.ones((1,len(path))),label,dict(gripper_pose=start,gripper_open=np.array(1.)),
                             dict(obstacle_centers=cs,obstacle_halfsizes=hs),cfg)
    assert c[0]['post_only_tip_valid'] and not c[0]['workspace_floor_correct'] and not c[0]['TipValid']
    assert metrics['TipValidAtK']==0 and metrics['original_post_only_metrics']['TipValidAtK']==1
