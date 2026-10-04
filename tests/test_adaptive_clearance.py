import torch
from routeset.adaptive_clearance import SegmentDual


def test_safe_zero_gradient_and_persistent_violation_strengthens():
    dual = SegmentDual(2, 1, 1, 100., .1, 'cpu')
    dual.weights.fill_(2)
    safe = torch.tensor([[[[1., 0., 0.], [2., 0., 0.]]]], requires_grad=True)
    c = torch.zeros(1, 1, 3);h = torch.ones(1, 1, 3)*.1
    loss, signed = dual.loss(safe, c, h, [0]);loss.backward()
    assert loss.item()==0 and safe.grad.abs().sum()==0
    dual.update([0], signed);assert dual.weights[0].item()<2
    unsafe = torch.tensor([[[[-.2, .05, 0.], [.2, .05, 0.]]]], requires_grad=True)
    first, signed = dual.loss(unsafe, c, h, [1]);dual.update([1], signed)
    second, _ = dual.loss(unsafe, c, h, [1])
    assert second>first and dual.weights[1].item()>2
    second.backward();assert unsafe.grad.abs().sum()>0


def test_repeated_requests_projection_and_exact_resume():
    dual = SegmentDual(2, 1, 1, 100., .1, 'cpu')
    dual.update([0,0,1], torch.tensor([[[.1]],[[.3]],[[-.1]]]))
    torch.testing.assert_close(dual.weights.flatten(), torch.tensor([.64,0.]))
    restored = SegmentDual(2, 1, 1, 10., 1., 'cpu')
    restored.load_state_dict(dual.state_dict())
    for x in (dual, restored): x.update([0,1], torch.tensor([[[100.]], [[100.]]]))
    assert torch.equal(dual.weights, restored.weights)
    assert dual.weights.max()==dual.cap
