import torch
from routeset.observed_probability import ProbabilisticGeometryRouteHead
from routeset.goal_preserving import GoalPreservingRouteHead, verify_gradient_routes
from routeset.segment_clearance import segment_clearance_loss


def test_endpoint_invariance_and_real_gradient_routing():
    torch.manual_seed(0)
    model = GoalPreservingRouteHead.from_baseline(ProbabilisticGeometryRouteHead(feature_dim=16, width=16, point_width=8, horizon=24, max_candidates=8))
    current = torch.zeros(2, 8)
    inputs = dict(features=torch.randn(2, 16), current=current, world_xyz=torch.rand(2, 20, 3)*.1+.2,
                  rgb=torch.rand(2, 20, 3), uv=torch.rand(2, 20, 2), depth=torch.ones(2, 20), valid_mask=torch.ones(2, 20, dtype=torch.bool))
    paths, _, details = model(**inputs)
    goal_loss = (details['goal']-.4).square().mean()
    clearance = segment_clearance_loss(model.clearance_paths(details, current), torch.ones(2, 1, 3)*.1, torch.ones(2, 1, 3)*.1)
    result = verify_gradient_routes(model, details, current, paths, clearance, goal_loss)
    assert result['clearance']['route'] > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.zero_grad();clearance.backward();optimizer.step()
    after, _, _ = model(**inputs)
    torch.testing.assert_close(paths[:, :, -1], after[:, :, -1], rtol=0, atol=0)
    assert not torch.equal(paths[:, :, 1:-1], after[:, :, 1:-1])
    torch.testing.assert_close(after[:, :, 0], current[:, None, :3].expand(-1, 8, -1), rtol=0, atol=0)


def test_reload_and_no_geometry_truth_forward_interface():
    torch.manual_seed(3)
    base = ProbabilisticGeometryRouteHead(feature_dim=16, width=16, point_width=8)
    a = GoalPreservingRouteHead.from_baseline(base)
    b = GoalPreservingRouteHead.from_baseline(base)
    b.load_state_dict(a.state_dict(), strict=True)
    assert set(a.state_dict()) == set(b.state_dict())
    import inspect
    assert set(inspect.signature(a.forward).parameters) == {'features','current','world_xyz','rgb','uv','depth','valid_mask','k'}
