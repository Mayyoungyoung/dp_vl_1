from types import SimpleNamespace
import pytest

from scripts.benchmark_demo_render import demo_camera_flags, select_train_parent
from scripts.observation_collect_multitask import registration


def scene():
    camera=SimpleNamespace(rgb=True,depth=True,point_cloud=False,mask=False)
    return SimpleNamespace(get_observation_config=lambda:SimpleNamespace(front_camera=camera)),camera


def test_on_mode_preserves_rgbd():
    item,camera=scene()
    with demo_camera_flags(item,False):assert camera.rgb and camera.depth
    assert camera.rgb and camera.depth


def test_off_mode_always_restores_configuration():
    item,camera=scene()
    with pytest.raises(RuntimeError):
        with demo_camera_flags(item,True):
            assert not camera.rgb and not camera.depth
            raise RuntimeError('demo failure')
    assert camera.rgb and camera.depth and not camera.point_cloud and not camera.mask


def test_unsupported_configuration_is_not_mutated():
    item,camera=scene();camera.point_cloud=True
    with pytest.raises(ValueError):
        with demo_camera_flags(item,True):pass
    assert camera.rgb and camera.depth and camera.point_cloud


def test_audit_parent_selection_is_registered_train_only():
    plan=registration(281000)
    assert select_train_parent(plan,'pick_and_lift_291000')['task']=='pick_and_lift'
    assert select_train_parent(plan,'slide_block_to_target_331000')['split']=='TRAIN'
    for parent in ('pick_and_lift_291016','reach_target_281022','missing_parent'):
        with pytest.raises(ValueError):select_train_parent(plan,parent)
