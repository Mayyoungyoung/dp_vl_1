from types import SimpleNamespace
import pytest

from scripts.benchmark_demo_render import demo_camera_flags


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
