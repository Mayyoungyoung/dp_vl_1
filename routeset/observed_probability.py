"""Observation-conditioned route reliability and balanced reference mass.

q consumes the ACTUAL path and observed geometry, never checker geometry.
No softmax competition between validity scores. pi is a separate distribution.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .observed_geometry import ObservedGeometryRouteHead


class ProbabilisticGeometryRouteHead(ObservedGeometryRouteHead):
    """Same observed backbone and whole-route queries, with separate mode mass.

    Returning fewer routes is a downstream selection, not another forward with
    fewer interacting queries. q is separately trained on this model's pools.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_mass = nn.Linear(self.head.queries.shape[-1], 1)
        nn.init.zeros_(self.mode_mass.weight)
        nn.init.zeros_(self.mode_mass.bias)

    def forward(self, features, current, world_xyz, rgb, uv, depth, valid_mask, k=None):
        k = self.head.max_candidates if k is None else k
        if k != self.head.max_candidates or self.refiner is not None or self.endpoint_mode != 'surface_anchor':
            raise ValueError('Fixed internal M and unchanged surface-anchor generator required')
        geometry = self.geometry(features,current,world_xyz,rgb,uv,depth,valid_mask)
        context = self.head.feature_encoder(features)+self.head.state_encoder(current)+geometry['context']
        tokens = context[:,None]+self.head.queries[None]
        for block in self.head.blocks:
            tokens = block(tokens,context)
        prediction = self.head.output(tokens).reshape(len(features),k,self.head.horizon-1,4)
        fractions = torch.linspace(0,1,self.head.horizon,device=features.device,dtype=features.dtype)[1:]
        line = current[:,None,None,:3]+fractions[None,None,:,None]*(geometry['anchor_xyz']-current[:,:3])[:,None,None]
        intermediate = line[:,:,:-1]+prediction[:,:,:-1,:3]
        endpoint = geometry['anchor_xyz'][:,None,None]+self.endpoint_residual_bound*prediction[:,:,-1:,:3].tanh()
        xyz = torch.cat([current[:,None,None,:3].expand(-1,k,1,-1),intermediate,endpoint],2)
        opened = torch.cat([current[:,None,None,7].expand(-1,k,1).clamp(0,1),prediction[...,3].sigmoid()],2)
        geometry['pi_logits'] = self.mode_mass(tokens).squeeze(-1)
        geometry['pi'] = geometry['pi_logits'].softmax(-1)
        return xyz,opened,geometry


def route_observation_features(paths, events, current, world_xyz, rgb, valid_mask,
                               point_features, context, anchor):
    """Query observed nearest surfaces at vertices and segment midpoints.

    Surface distance is an observation feature, not a collision certificate.
    Preserves free-space paths and masks invalid depth points BEFORE distance.
    """
    if paths.ndim != 4 or paths.shape[-1] != 3 or events.shape != paths.shape[:-1]:
        raise ValueError('Expected B,M,H,3 paths and B,M,H events')
    if not torch.isfinite(paths).all() or not valid_mask.any(-1).all():
        raise ValueError('Finite routes and nonempty observed point clouds required')
    b, m, h, _ = paths.shape
    mids = (paths[:, :, 1:] + paths[:, :, :-1]) / 2
    probes = torch.cat([paths, mids], 2)
    event = torch.cat([events, (events[:, :, 1:]+events[:, :, :-1])/2], 2)
    direction = torch.cat([torch.zeros_like(paths[:, :, :1]), paths[:, :, 1:]-paths[:, :, :-1]], 2)
    direction = torch.cat([direction, paths[:, :, 1:]-paths[:, :, :-1]], 2)
    safe = torch.where(valid_mask[..., None], world_xyz, torch.zeros_like(world_xyz))
    distance = torch.cdist(probes.reshape(b, -1, 3), safe).masked_fill(~valid_mask[:, None], float('inf'))
    values, indices = distance.topk(min(4, world_xyz.shape[1]), dim=-1, largest=False)
    if not torch.isfinite(values).all():
        raise ValueError('At least four valid observed points required')
    batches = torch.arange(b, device=paths.device)[:, None, None]
    nearby_xyz = safe[batches, indices].mean(2).reshape(b, m, 2*h-1, 3)
    nearby_rgb = rgb[batches, indices].mean(2).reshape(b, m, 2*h-1, 3)
    nearby_features = point_features[batches, indices].mean(2).reshape(b, m, 2*h-1, -1)
    d = values[..., [0, -1]].reshape(b, m, 2*h-1, 2)
    node = torch.cat([probes-current[:, None, None, :3], direction,
                      nearby_xyz-probes, nearby_rgb, d, event[..., None], nearby_features], -1)
    fraction = torch.cat([torch.linspace(0, 1, h, device=paths.device),
                          (torch.arange(h-1, device=paths.device)+.5)/(h-1)])
    node = torch.cat([node, fraction[None, None, :, None].expand(b, m, -1, -1)], -1)
    route_context = torch.cat([context[:, None].expand(-1, m, -1),
                              paths[:, :, -1]-anchor[:, None]], -1)
    return node, route_context


class RouteValidityHead(nn.Module):
    def __init__(self, node_dim=80, context_dim=131, width=64):
        super().__init__()
        self.nodes = nn.Sequential(nn.Linear(node_dim, width), nn.SiLU(),
                                   nn.Linear(width, width), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(2*width+context_dim, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, nodes, context):
        encoded = self.nodes(nodes)
        return self.score(torch.cat([encoded.mean(-2), encoded.amax(-2), context], -1)).squeeze(-1)


def select_route_indices(paths, q, k, diversity_distance=.04):
    """q-first selection; prefer distinct routes among q>=.5, then fill by q.

    The fixed .5 threshold is a presentation rule, not a validity certificate.
    All internal candidates and their original pi/q remain reportable.
    """
    if not 1 <= k <= paths.shape[1]:
        raise ValueError('Requested K must not exceed generated M')
    selected=[]
    for b in range(len(paths)):
        order=torch.argsort(q[b],descending=True,stable=True).tolist()
        chosen=[order[0]]
        for i in order[1:]:
            if len(chosen)==k:break
            if q[b,i]>=.5 and all(torch.linalg.vector_norm(paths[b,i]-paths[b,j],dim=-1).mean()>=diversity_distance for j in chosen):
                chosen.append(i)
        for i in order:
            if len(chosen)==k:break
            if i not in chosen:chosen.append(i)
        selected.append(chosen)
    return torch.tensor(selected,device=paths.device)


class ScoredRoutePlanner(nn.Module):
    """Deployment interface: observed inputs -> M routes/pi/q and selected K."""
    def __init__(self,generator,scorer,normalization,temperature=1.):
        super().__init__()
        if temperature<=0:raise ValueError('Positive calibration temperature required')
        self.generator=generator;self.scorer=scorer;self.temperature=float(temperature)
        for key,value in normalization.items():self.register_buffer(key,torch.as_tensor(value))

    def forward(self,features,current,world_xyz,rgb,uv,depth,valid_mask,return_k=1):
        inputs=dict(features=features,current=current,world_xyz=world_xyz,rgb=rgb,uv=uv,depth=depth,valid_mask=valid_mask)
        paths,events,details=self.generator(**inputs)
        geometry=self.generator.geometry(**inputs,return_point_features=True)
        context=self.generator.head.feature_encoder(features)+self.generator.head.state_encoder(current)+geometry['context']
        nodes,ctx=route_observation_features(paths,events,current,world_xyz,rgb,valid_mask,geometry['point_features'],context,geometry['anchor_xyz'])
        logits=self.scorer((nodes-self.nodes_mean)/self.nodes_std,(ctx-self.context_mean)/self.context_std)
        q=(logits/self.temperature).sigmoid()
        pi=details.get('pi')
        indices=select_route_indices(paths,q,return_k)
        return dict(paths=paths,events=events,pi=pi,q=q,selected_indices=indices,
                    internal_candidates=paths.shape[1],returned_candidates=return_k)


def reference_cluster_weights(paths, mask, threshold=.04):
    """Deterministic complete-link geometric grouping; no manual type labels.

    Each discovered cluster has equal mass; samples within it share the mass.
    This is a reference approximation, not inferred homotopy or exhaustive modes.
    """
    paths, mask = np.asarray(paths), np.asarray(mask, dtype=bool)
    weights = np.zeros(mask.shape, dtype=np.float32)
    for b in range(len(paths)):
        ids = np.flatnonzero(mask[b])
        if not len(ids):
            raise ValueError('At least one positive reference required')
        # Canonical geometry order makes grouping independent of input ordering.
        ids = sorted(ids, key=lambda i: tuple(paths[b, i].reshape(-1)))
        groups = []
        for i in ids:
            for group in groups:
                if all(np.linalg.norm(paths[b, i]-paths[b, j], axis=-1).mean() <= threshold for j in group):
                    group.append(i)
                    break
            else:
                groups.append([i])
        for group in groups:
            weights[b, group] = 1 / (len(groups)*len(group))
    return weights


def balanced_assignment_loss(prediction, target, weights):
    """Reference-to-hypothesis WTA coverage plus hypothesis-to-positive fit.

    References choose one whole route; no coordinatewise mixture. pi targets
    are detached responsibilities with ties shared across duplicate hypotheses.
    """
    cost = (prediction[:, :, None]-target[:, None]).square().mean((-1, -2))
    valid = weights > 0
    masked = cost.masked_fill(~valid[:, None], float('inf'))
    fit = masked.min(2).values.mean()
    cover = (cost.min(1).values * weights).sum(1).mean()
    with torch.no_grad():
        best = cost.min(1, keepdim=True).values
        ties = torch.isclose(cost, best, atol=1e-8, rtol=1e-5).to(cost.dtype)
        responsibility = ties / ties.sum(1, keepdim=True).clamp_min(1)
        mass = (responsibility * weights[:, None]).sum(2)
    return (fit+cover)/2, mass


def selection_metrics(logits, labels, paths, parents, temperature=1.):
    logits, labels = np.asarray(logits), np.asarray(labels, dtype=float)
    p = 1/(1+np.exp(-np.clip(logits/temperature, -60, 60)))
    pick = logits.argmax(1)
    shortest = np.linalg.norm(np.diff(paths, axis=2), axis=-1).sum(2).argmin(1)
    row = np.arange(len(labels))
    selected = labels[row, pick]
    result = dict(requests=len(labels), candidates=labels.size, parent_count=len(set(parents)),
                  valid=float(labels.mean()), any_valid=float(labels.max(1).mean()),
                  selected_valid=float(selected.mean()), first_valid=float(labels[:, 0].mean()),
                  random_expected_valid=float(labels.mean()), shortest_valid=float(labels[row, shortest].mean()),
                  brier=float(((p-labels)**2).mean()),
                  selected_brier=float(((p[row, pick]-selected)**2).mean()))
    bins = []
    for lo in np.arange(0, 1, .1):
        keep = (p >= lo) & (p < lo+.1 if lo < .9 else p <= 1)
        bins.append(dict(lower=float(lo), n=int(keep.sum()), confidence=float(p[keep].mean()) if keep.any() else None,
                         valid=float(labels[keep].mean()) if keep.any() else None))
    result['reliability_bins'] = bins
    parent_rows = []
    for parent in sorted(set(parents)):
        keep = parents == parent
        parent_rows.append(dict(parent_id=str(parent), q=float(selected[keep].mean()),
                                first=float(labels[keep, 0].mean()), random=float(labels[keep].mean())))
    result['per_parent'] = parent_rows
    rng = np.random.default_rng(78123)
    delta = np.array([r['q']-r['random'] for r in parent_rows])
    boot = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(1)
    result['q_minus_random_parent_bootstrap95'] = np.quantile(boot, [.025, .975]).tolist()
    return result
