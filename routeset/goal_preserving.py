"""Observation-only goal/interior factorization with strict clearance routing."""
import copy
import torch
from torch import nn
from .observed_probability import ProbabilisticGeometryRouteHead


class GoalPreservingRouteHead(ProbabilisticGeometryRouteHead):
    @classmethod
    def from_baseline(cls, baseline):
        # Preserve the historical initialization before converting coordinates.
        model = copy.deepcopy(baseline)
        model.__class__ = cls
        model.goal_queries = nn.Parameter(model.head.queries.detach().clone())
        model.goal_blocks = copy.deepcopy(model.head.blocks)
        model.goal_output = copy.deepcopy(model.head.output)
        old = model.goal_output[-1]
        end = nn.Linear(old.in_features, 3, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            end.weight.copy_(old.weight[-4:-1]); end.bias.copy_(old.bias[-4:-1])
        model.goal_output[-1] = end
        # Convert old interior offsets to deformation coordinates. This retains
        # their initial magnitude; it is not an extra trained warm-up or repair.
        with torch.no_grad():
            u = torch.linspace(0, 1, model.head.horizon, device=old.weight.device)[1:-1]
            phi = 4*u*(1-u)
            w = model.head.output[-1].weight.view(model.head.horizon-1, 4, -1)
            b = model.head.output[-1].bias.view(model.head.horizon-1, 4)
            w[:-1, :3].div_(phi[:, None, None]); b[:-1, :3].div_(phi[:, None])
        return model

    def decode(self, context, current, goal):
        tokens = context[:, None]+self.head.queries[None]
        for block in self.head.blocks:
            tokens = block(tokens, context)
        pred = self.head.output(tokens).reshape(len(context), self.head.max_candidates, self.head.horizon-1, 4)
        u = torch.linspace(0, 1, self.head.horizon, device=context.device, dtype=context.dtype)[1:]
        line = current[:, None, None, :3]+u[None, None, :, None]*(goal-current[:, :3])[:, None, None]
        xyz = line+(4*u*(1-u))[None, None, :, None]*pred[..., :3]
        xyz = torch.cat([current[:, None, None, :3].expand(-1, self.head.max_candidates, 1, -1), xyz], 2)
        events = torch.cat([current[:, None, None, 7].expand(-1, self.head.max_candidates, 1).clamp(0, 1), pred[..., 3].sigmoid()], 2)
        return xyz, events, tokens

    def forward(self, features, current, world_xyz, rgb, uv, depth, valid_mask, k=None):
        if k not in (None, self.head.max_candidates):
            raise ValueError('Fixed M candidates required')
        geometry = self.geometry(features, current, world_xyz, rgb, uv, depth, valid_mask)
        context = self.head.feature_encoder(features)+self.head.state_encoder(current)+geometry['context']
        tokens = context[:, None]+self.goal_queries[None]
        for block in self.goal_blocks:
            tokens = block(tokens, context)
        goal = geometry['anchor_xyz']+self.endpoint_residual_bound*self.goal_output(tokens).tanh().mean(1)
        xyz, events, tokens = self.decode(context, current, goal)
        geometry.update(goal=goal, route_context=context, pi_logits=self.mode_mass(tokens).squeeze(-1))
        geometry['pi'] = geometry['pi_logits'].softmax(-1)
        return xyz, events, geometry

    def clearance_paths(self, details, current):
        # Also cut the shared encoders: detaching goal alone does not protect
        # goal grounding from clearance updates through shared context.
        return self.decode(details['route_context'].detach(), current.detach(), details['goal'].detach())[0]


def verify_gradient_routes(model, details, current, normal_paths, clearance_loss, goal_loss):
    named = list(model.named_parameters())
    result = {}
    for label, loss in [('clearance', clearance_loss), ('goal', goal_loss)]:
        gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
        norms = {'goal': 0., 'encoder': 0., 'route': 0.}
        for (name, _), grad in zip(named, gradients):
            group = 'goal' if name.startswith('goal_') else ('encoder' if name.startswith(('geometry.', 'head.feature_encoder.', 'head.state_encoder.')) else 'route')
            if grad is not None: norms[group] += float(grad.detach().square().sum())
        result[label] = {key: value**.5 for key, value in norms.items()}
    assert result['clearance']['goal'] == result['clearance']['encoder'] == 0
    assert result['clearance']['route'] > 0
    assert result['goal']['route'] == 0 and result['goal']['goal'] > 0
    routed = model.clearance_paths(details, current)
    torch.testing.assert_close(routed, normal_paths, rtol=0, atol=0)
    assert torch.equal(normal_paths[:, :, -1], details['goal'][:, None].expand(-1, model.head.max_candidates, -1)) or torch.allclose(normal_paths[:, :, -1], details['goal'][:, None], atol=1e-7)
    result['identical_forward_paths'] = True
    return result
