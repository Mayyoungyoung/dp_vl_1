"""Route-conditioned task/feasibility factors; no truth is a forward input."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .observed_probability import RouteValidityHead


ARMS = ('single', 'joint', 'marginal', 'conditional', 'conditional_endpoint')


def factor_labels(candidates):
    task = np.asarray([c['semantic_goal_correct'] and c['finite_event_values']
                       and c['event_state_sequence_correct'] for c in candidates], dtype=bool)
    feas = np.asarray([c['finite_xyz'] and c['starts_at_current_state']
                       and c['tip_segments_clear'] for c in candidates], dtype=bool)
    joint = np.asarray([c['TipValid'] for c in candidates], dtype=bool)
    if not np.array_equal(task & feas, joint):
        raise ValueError('Factor conjunction does not reproduce original checker')
    return task, feas


class FactorRouteScorer(nn.Module):
    def __init__(self, arm='conditional', width=64):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.heads = nn.ModuleList([RouteValidityHead(width=width)
                                   for _ in range(1 if arm == 'single' else 2)])

    def forward(self, nodes, context):
        if self.arm == 'conditional_endpoint':
            # T depends on endpoint and constant-reach event correctness, not
            # the intermediate spatial route. Incoming endpoint direction is
            # replaced by start/min/max event values (same80 input dimensions).
            task_nodes=nodes[...,23:24,:].clone()
            event=nodes[...,14]
            task_nodes[...,0,3:6]=torch.stack([event[...,0],event.amin(-1),event.amax(-1)],-1)
            return torch.stack([self.heads[0](task_nodes,context),self.heads[1](nodes,context)],-1)
        return torch.stack([head(nodes, context) for head in self.heads], -1)


def log_joint(logits, arm):
    if arm in ('marginal', 'conditional', 'conditional_endpoint'):
        return F.logsigmoid(logits).sum(-1)
    z = logits[..., 0] if arm == 'single' else logits.sum(-1) / math.sqrt(2.)
    return F.logsigmoid(z)


def joint_nll(logits, labels, arm):
    lp = log_joint(logits, arm).clamp(max=-1e-7)
    ln = torch.log(-torch.expm1(lp))
    return -(labels * lp + (1-labels) * ln).mean()


def training_loss(logits, task, feas, arm):
    if arm in ('single', 'joint'):
        return joint_nll(logits, task * feas, arm)
    lt = F.binary_cross_entropy_with_logits(logits[..., 0], task)
    lf = F.binary_cross_entropy_with_logits(logits[..., 1], feas, reduction='none')
    if arm in ('conditional', 'conditional_endpoint'):
        lf = (lf * task).sum() / task.sum().clamp_min(1)
    else:
        lf = lf.mean()
    return lt + lf


def apply_calibration(logits, arm, calibration):
    if arm in ('marginal', 'conditional', 'conditional_endpoint'):
        temperature = torch.as_tensor(calibration['temperatures'], device=logits.device, dtype=logits.dtype)
        factors = (logits / temperature).sigmoid()
        return factors.prod(-1), factors
    z = logits[..., 0] if arm == 'single' else logits.sum(-1) / math.sqrt(2.)
    return (calibration['slope'] * z + calibration['intercept']).sigmoid(), None


class FactoredRoutePlanner(nn.Module):
    def __init__(self, generator, scorer, normalization, calibration):
        super().__init__()
        self.generator, self.scorer = generator, scorer
        self.calibration = calibration
        for k, v in normalization.items():
            self.register_buffer(k, torch.as_tensor(v))

    def forward(self, features, current, world_xyz, rgb, uv, depth, valid_mask, return_k=1):
        from .observed_probability import route_observation_features, select_route_indices
        inp = dict(features=features, current=current, world_xyz=world_xyz, rgb=rgb,
                   uv=uv, depth=depth, valid_mask=valid_mask)
        paths, events, _ = self.generator(**inp)
        geo = self.generator.geometry(**inp, return_point_features=True)
        context = self.generator.head.feature_encoder(features) + self.generator.head.state_encoder(current) + geo['context']
        nodes, ctx = route_observation_features(paths, events, current, world_xyz, rgb,
                                               valid_mask, geo['point_features'], context, geo['anchor_xyz'])
        logits = self.scorer((nodes-self.nodes_mean)/self.nodes_std,
                             (ctx-self.context_mean)/self.context_std)
        q, factors = apply_calibration(logits, self.scorer.arm, self.calibration)
        indices = select_route_indices(paths, q, return_k)
        rows = torch.arange(len(paths), device=paths.device)[:, None]
        out = dict(paths=paths, events=events, q=q, selected_indices=indices,
                   selected_paths=paths[rows, indices], selected_q=q[rows, indices])
        if factors is not None:
            out.update(q_task=factors[..., 0], q_feas=factors[..., 1])
        return out


def load_factored_planner(path, device='cpu'):
    from .observed_probability import ProbabilisticGeometryRouteHead
    data = torch.load(path, map_location='cpu', weights_only=False)
    generator = ProbabilisticGeometryRouteHead(**data['head_options'])
    scorer = FactorRouteScorer(data['arm'], data['width'])
    model = FactoredRoutePlanner(generator, scorer, data['normalization'], data['calibration'])
    model.load_state_dict(data['model'], strict=True)
    return model.to(device).eval()
