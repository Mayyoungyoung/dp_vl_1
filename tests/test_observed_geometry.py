import numpy as np
import pytest
import torch
from routeset.observed_geometry import ObservedGeometryEncoder, ObservedGeometryRouteHead, backproject_rgbd


def test_signed_intrinsics_and_camera_world_roundtrip():
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    depth = np.full((6, 8), 2.0)
    K = np.array([[-4., .3, 4.], [0., -5., 3.], [0., 0., 1.]])
    E = np.eye(4)
    E[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    E[:3, 3] = [1, 2, 3]
    points = backproject_rgbd(rgb, depth, K, E, pixel_stride=2)
    camera = (points['world_xyz']-E[:3, 3]) @ E[:3, :3]
    projected = camera @ K.T
    np.testing.assert_allclose(projected[:, :2]/projected[:, 2:], points['pixel_xy'], atol=1e-6)
    np.testing.assert_allclose(camera[:, 2], 2., atol=1e-6)


def test_invalid_depth_stays_masked_and_empty_fails():
    depth = np.array([[1., np.nan], [0., 30.]])
    result = backproject_rgbd(np.zeros((2, 2, 3), np.uint8), depth, np.eye(3), np.eye(4), pixel_stride=1)
    assert result['valid_mask'].tolist() == [True, False, False, False]
    assert np.isfinite(result['world_xyz']).all()
    with pytest.raises(ValueError, match='no valid'):
        backproject_rgbd(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2)), np.eye(3), np.eye(4))


def inputs():
    torch.manual_seed(12)
    return dict(features=torch.randn(2, 16), current=torch.randn(2, 8), world_xyz=torch.randn(2, 7, 3),
                rgb=torch.rand(2, 7, 3), uv=torch.rand(2, 7, 2)*2-1, depth=torch.ones(2, 7),
                valid_mask=torch.tensor([[True]*6+[False], [True]*5+[False]*2]))


def test_mask_permutation_and_surface_anchor():
    model = ObservedGeometryEncoder(16, width=12, point_width=8)
    batch = inputs()
    out = model(**batch)
    assert out['context'].shape == (2, 12)
    assert torch.equal(out['attention'][~batch['valid_mask']], torch.zeros(3))
    assert torch.allclose(out['attention'].sum(1), torch.ones(2))
    permutation = torch.tensor([6, 2, 1, 3, 0, 5, 4])
    permuted = {key: (value[:, permutation] if key not in ['features', 'current'] else value) for key,value in batch.items()}
    permuted_out = model(**permuted)
    assert torch.allclose(out['context'], permuted_out['context'], atol=1e-6)
    assert torch.allclose(out['anchor_xyz'], permuted_out['anchor_xyz'], atol=1e-6)
    changed = {key:value.clone() for key,value in batch.items()}
    changed['world_xyz'][~changed['valid_mask']] = float('nan')
    assert torch.allclose(model(**changed)['context'], out['context'], atol=1e-6)


def test_gradients_reach_task_and_geometry_without_labels():
    model = ObservedGeometryEncoder(16, width=12, point_width=8)
    batch = inputs()
    batch['features'].requires_grad_()
    batch['rgb'].requires_grad_()
    result = model(**batch)
    (result['context'].square().mean()+result['anchor_xyz'].square().mean()).backward()
    assert batch['features'].grad.abs().sum() > 0
    assert batch['rgb'].grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    with pytest.raises(TypeError):
        model(**batch, target_xyz=torch.zeros(2, 3))


def test_empty_point_mask_rejected():
    batch = inputs()
    batch['valid_mask'][0] = False
    with pytest.raises(ValueError, match='at least one valid'):
        ObservedGeometryEncoder(16)(**batch)


def test_route_head_start_endpoint_bound_and_parameter_update():
    torch.set_num_threads(1)
    model = ObservedGeometryRouteHead(16, horizon=8, width=16, point_width=8)
    batch = inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    xyz, opened, details = model(**batch)
    assert xyz.shape == (2, 4, 8, 3) and opened.shape == (2, 4, 8)
    assert torch.equal(xyz[:, 0, 0], batch['current'][:, :3])
    assert (xyz[:, :, -1]-details['anchor_xyz'][:, None]).abs().max() <= .05
    loss = xyz.square().mean() + opened.square().mean()
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    optimizer.step()
    assert any(not torch.equal(parameter, before[name]) for name,parameter in model.named_parameters() if name.startswith('geometry.point_encoder'))
    assert any(not torch.equal(parameter, before[name]) for name,parameter in model.named_parameters() if name.startswith('geometry.task_query'))
    assert any(not torch.equal(parameter, before[name]) for name,parameter in model.named_parameters() if name.startswith('head.output'))
