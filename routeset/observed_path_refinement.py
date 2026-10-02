"""One observation-only draft update, with matched local/global pooling.

Conventional route-head repair, not a new allocation or set mechanism. Both
modes instantiate identical parameters and materialize K drafts plus K finals.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def refinement_config(config):
    return dict(refinement_mode=config.get('refinement_mode','none'),
                refinement_sigma=config.get('refinement_sigma'),
                refinement_prefix_fraction=config.get('refinement_prefix_fraction'),
                refinement_bound=config.get('refinement_bound'))


def validate_refinement_resume(current, saved):
    if refinement_config(current) != refinement_config(saved):
        raise ValueError('resume config mismatch: observation refinement')


class ObservedPathRefiner(nn.Module):
    def __init__(self, point_width, mode, sigma, prefix_fraction, bound, hidden_width=64):
        super().__init__()
        if mode not in ('local','global'):
            raise ValueError('refinement mode must be local or global')
        if any(value is None or not math.isfinite(value) or value <= 0 for value in (sigma,prefix_fraction,bound)) or prefix_fraction > 1:
            raise ValueError('explicit positive TRAIN-prespecified sigma/prefix/bound required; prefix <= 1')
        self.mode, self.sigma, self.prefix_fraction, self.bound = mode, sigma, prefix_fraction, bound
        # point feature + observed centroid relative to query + query relative
        # to current state + draft tangent + draft normalized arc position.
        self.update = nn.Sequential(nn.Linear(point_width+10,hidden_width),nn.SiLU(),nn.Linear(hidden_width,3))
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, draft, point_features, world_xyz, valid_mask, current):
        if draft.ndim != 4 or draft.shape[-1] != 3 or draft.shape[2] < 3:
            raise ValueError('finite draft [B,K,H>=3,3] required')
        batch, candidates, horizon, _ = draft.shape
        if (world_xyz.ndim != 3 or world_xyz.shape[0] != batch or world_xyz.shape[-1] != 3 or
                point_features.shape[:2] != world_xyz.shape[:2] or valid_mask.shape != world_xyz.shape[:2] or
                current.shape != (batch,8)):
            raise ValueError('matching current and observed points required')
        valid_mask = valid_mask.bool()
        if not bool(valid_mask.any(1).all()) or not bool(torch.isfinite(draft).all()):
            raise ValueError('finite draft and at least one valid observed point required')
        safe_xyz = torch.where(valid_mask[...,None],world_xyz,torch.zeros_like(world_xyz))
        safe_features = torch.where(valid_mask[...,None],point_features,torch.zeros_like(point_features))
        if not bool(torch.isfinite(safe_xyz).all() and torch.isfinite(safe_features).all()):
            raise ValueError('valid observed features must be finite')
        query = draft[:,:,1:-1].reshape(batch,-1,3)
        if self.mode == 'local':
            # B x (K*(H-2)) x N, without expanding the N x point_width features
            # per waypoint. Invalid points cannot influence weights or moments.
            distance2 = (query.square().sum(-1,keepdim=True)+safe_xyz.square().sum(-1)[:,None]
                         -2*torch.bmm(query,safe_xyz.transpose(1,2))).clamp_min(0)
            logits = -distance2/(2*self.sigma**2)
            weights = logits.masked_fill(~valid_mask[:,None],float('-inf')).softmax(-1)
            pooled_features = torch.bmm(weights,safe_features)
            centroid = torch.bmm(weights,safe_xyz)
        else:
            weights = valid_mask.to(draft.dtype)/valid_mask.sum(1,keepdim=True)
            pooled_features = (weights[...,None]*safe_features).sum(1)[:,None].expand(-1,query.shape[1],-1)
            centroid = (weights[...,None]*safe_xyz).sum(1)[:,None].expand(-1,query.shape[1],-1)
        lengths = torch.linalg.vector_norm(draft[:,:,1:]-draft[:,:,:-1],dim=-1)
        prefix_arc = lengths.cumsum(-1)[:,:,:-1]/lengths.sum(-1,keepdim=True).clamp_min(1e-8)
        eligible = prefix_arc <= self.prefix_fraction
        tangent = F.normalize(draft[:,:,2:]-draft[:,:,:-2],dim=-1,eps=1e-8).reshape(batch,-1,3)
        features = torch.cat([pooled_features,centroid-query,query-current[:,None,:3],tangent,
                              prefix_arc.reshape(batch,-1,1)],dim=-1)
        delta = self.bound*self.update(features).tanh().reshape(batch,candidates,horizon-2,3)
        delta = torch.where(eligible[...,None],delta,torch.zeros_like(delta))
        refined = torch.cat([draft[:,:,:1],draft[:,:,1:-1]+delta,draft[:,:,-1:]],dim=2)
        return refined, dict(refinement_eligible=eligible,refinement_delta=delta,
                             refinement_prefix_arc=prefix_arc)
