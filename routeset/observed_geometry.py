"""Observation-only RGB-D geometry and task-conditioned spatial aggregation.

This is a conventional grounding baseline component, not a novelty claim.
Camera intrinsics retain their supplied signs (PyRep uses negative focal lengths)
and extrinsics map camera coordinates to world coordinates. The API takes no
target coordinates, object geometry, segmentation labels or future trajectory.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def positive_endpoint_attention_loss(attention, world_xyz, valid_mask,
                                     reference_endpoints, reference_mask, sigma=.025):
    """Train-only spatial target from recorded positive endpoint observations.

    This auxiliary grounding loss supplies no endpoint to the model forward.
    Multiple positive endpoints form a mixture, never their possibly invalid
    average. It is not an existence/validity label for unmatched route outputs.
    """
    if sigma <= 0 or not bool(reference_mask.any(1).all()):
        raise ValueError('positive sigma and at least one positive endpoint per training example required')
    if attention.shape != valid_mask.shape or world_xyz.shape[:2] != attention.shape:
        raise ValueError('matching observed point attention/coordinates/mask required')
    if not bool(valid_mask.any(1).all()):
        raise ValueError('each training observation needs valid depth')
    with torch.no_grad():
        distance2 = ((world_xyz[:, :, None]-reference_endpoints[:, None])**2).sum(-1)
        kernels = (-distance2/(2*sigma**2)).masked_fill(~reference_mask[:, None], float('-inf'))
        logits = torch.logsumexp(kernels, dim=-1).masked_fill(~valid_mask, float('-inf'))
        positive_density = logits.softmax(-1)
    return -(positive_density*attention.clamp_min(1e-12).log()).sum(-1).mean()


def backproject_rgbd(rgb, depth, intrinsics, camera_to_world, pixel_stride=2, max_depth=10.0):
    """Build an unlabelled observed point grid, preserving exact sampled pixels.

    Depth is optical-axis depth in metres, matching RLBench depth_in_meters.
    No vertical flip or absolute-value focal correction is applied. Invalid
    depth samples remain masked; no occluded geometry is filled in.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    intrinsics, camera_to_world = np.asarray(intrinsics), np.asarray(camera_to_world)
    if depth.ndim != 2 or rgb.shape != depth.shape + (3,):
        raise ValueError('matching depth [H,W] and RGB [H,W,3] are required')
    if intrinsics.shape != (3, 3) or camera_to_world.shape not in ((3, 4), (4, 4)):
        raise ValueError('K [3,3] and camera-to-world [3,4] or [4,4] required')
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_world).all():
        raise ValueError('nonfinite camera calibration')
    if pixel_stride < 1 or not isinstance(pixel_stride, int) or max_depth <= 0:
        raise ValueError('positive integer pixel_stride and positive max_depth required')
    if abs(np.linalg.det(intrinsics)) < 1e-12:
        raise ValueError('singular intrinsics')
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(pixel_stride // 2, h, pixel_stride),
                         np.arange(pixel_stride // 2, w, pixel_stride), indexing='ij')
    if not xx.size:
        raise ValueError('pixel stride leaves no observed points')
    sampled_depth = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0) & (sampled_depth <= max_depth)
    if not valid.any():
        raise ValueError('no valid observed depth points')
    safe_depth = np.where(valid, sampled_depth, 0.)
    pixel_homogeneous = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    rays = pixel_homogeneous @ np.linalg.inv(intrinsics).T
    camera_xyz = rays * safe_depth[..., None]
    world_xyz = camera_xyz @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    sampled_rgb = rgb[yy, xx].astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        sampled_rgb /= 255.
    if not np.isfinite(sampled_rgb).all() or sampled_rgb.min() < 0 or sampled_rgb.max() > 1:
        raise ValueError('RGB must be uint8 or finite float in [0,1]')
    normalized_uv = np.stack([(xx + .5) / w * 2 - 1, (yy + .5) / h * 2 - 1], axis=-1)
    return {'world_xyz': world_xyz.reshape(-1, 3).astype(np.float32),
            'rgb': sampled_rgb.reshape(-1, 3),
            'uv': normalized_uv.reshape(-1, 2).astype(np.float32),
            'depth': safe_depth.reshape(-1).astype(np.float32),
            'valid_mask': valid.reshape(-1),
            'pixel_xy': np.stack([xx, yy], -1).reshape(-1, 2).astype(np.int64)}


class ObservedGeometryEncoder(nn.Module):
    """Small Qwen-conditioned attention over RGB and observed metric points.

    `anchor_xyz` is the attention-weighted observed surface location. It is a
    learned estimate, not a supplied goal and not a certified endpoint. The
    route head may use context and anchor, but all baseline/mechanism variants
    must share this information. No point-selection oracle or repair is used.
    """
    def __init__(self, feature_dim=4096, width=128, point_width=64):
        super().__init__()
        self.feature_dim, self.width, self.point_width = feature_dim, width, point_width
        self.task_query = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, point_width))
        self.state_query = nn.Sequential(nn.Linear(8, point_width), nn.SiLU(), nn.Linear(point_width, point_width))
        # Relative world xyz3 + RGB3 + pixel uv2 + observed optical depth1.
        self.point_encoder = nn.Sequential(nn.Linear(9, point_width), nn.SiLU(), nn.Linear(point_width, point_width))
        self.log_attention_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.fusion = nn.Sequential(nn.Linear(point_width * 3 + 3, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, features, current, world_xyz, rgb, uv, depth, valid_mask):
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError('features must be [B,feature_dim]')
        batch = len(features)
        if current.shape != (batch, 8) or world_xyz.ndim != 3 or world_xyz.shape[0] != batch or world_xyz.shape[-1] != 3:
            raise ValueError('current [B,8] and observed world_xyz [B,N,3] required')
        points = world_xyz.shape[1]
        if rgb.shape != world_xyz.shape or uv.shape != (batch, points, 2) or depth.shape != (batch, points) or valid_mask.shape != (batch, points):
            raise ValueError('observed point fields must have matching shapes')
        valid_mask = valid_mask.bool()
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError('each observation must contain at least one valid point')
        relative_xyz = world_xyz - current[:, None, :3]
        raw = torch.cat([relative_xyz, rgb * 2 - 1, uv, depth[..., None]], dim=-1)
        # Prevent masked NaNs from propagating through the point MLP/softmax.
        raw = torch.where(valid_mask[..., None], raw, torch.zeros_like(raw))
        if not bool(torch.isfinite(raw).all()):
            raise ValueError('valid point features must be finite')
        point_features = self.point_encoder(raw)
        query = self.task_query(features) + self.state_query(current)
        logits = (F.normalize(point_features, dim=-1) * F.normalize(query, dim=-1)[:, None]).sum(-1)
        logits = logits * self.log_attention_scale.exp().clamp(max=100.)
        weights = logits.masked_fill(~valid_mask, float('-inf')).softmax(-1)
        safe_xyz = torch.where(valid_mask[..., None], world_xyz, torch.zeros_like(world_xyz))
        anchor = (weights[..., None] * safe_xyz).sum(1)
        attended = (weights[..., None] * point_features).sum(1)
        mean = (point_features * valid_mask[..., None]).sum(1) / valid_mask.sum(1, keepdim=True)
        context = self.fusion(torch.cat([attended, mean, query, anchor-current[:, :3]], dim=-1))
        return {'context': context, 'anchor_xyz': anchor, 'attention': weights}

    def active_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


class ObservedGeometryRouteHead(nn.Module):
    """Ordinary set regression with observed spatial context and learned endpoint.

    Uses the existing route-head modules unchanged. The geometric anchor is
    predicted from pixels, and the endpoint can move by a bounded learned
    residual. No route/target label enters this forward call.
    """
    def __init__(self, feature_dim, horizon=24, max_candidates=4, width=128,
                 depth=2, point_width=64, endpoint_residual_bound=.05,
                 geometry_pooling='spatial'):
        super().__init__()
        from .observed_route_head import ObservedRouteHead
        if geometry_pooling != 'spatial':
            raise NotImplementedError('only spatial pooling is currently implemented')
        if endpoint_residual_bound <= 0:
            raise ValueError('endpoint residual bound must be positive')
        self.geometry = ObservedGeometryEncoder(feature_dim, width, point_width)
        self.head = ObservedRouteHead(feature_dim, horizon, max_candidates, width, depth)
        self.endpoint_residual_bound = endpoint_residual_bound

    def forward(self, features, current, world_xyz, rgb, uv, depth, valid_mask, k=None):
        k = self.head.max_candidates if k is None else k
        if not 1 <= k <= self.head.max_candidates:
            raise ValueError('invalid candidate budget')
        geometry = self.geometry(features, current, world_xyz, rgb, uv, depth, valid_mask)
        context = self.head.feature_encoder(features) + self.head.state_encoder(current) + geometry['context']
        tokens = context[:, None] + self.head.queries[:k][None]
        for block in self.head.blocks:
            tokens = block(tokens, context)
        prediction = self.head.output(tokens).reshape(len(features), k, self.head.horizon-1, 4)
        fractions = torch.linspace(0, 1, self.head.horizon, device=features.device, dtype=features.dtype)[1:]
        reference_line = current[:, None, None, :3] + fractions[None, None, :, None] * (geometry['anchor_xyz']-current[:, :3])[:, None, None]
        intermediate = reference_line[:, :, :-1] + prediction[:, :, :-1, :3]
        endpoint = geometry['anchor_xyz'][:, None, None] + self.endpoint_residual_bound * prediction[:, :, -1:, :3].tanh()
        first = current[:, None, None, :3].expand(-1, k, 1, -1)
        xyz = torch.cat([first, intermediate, endpoint], dim=2)
        first_open = current[:, None, None, 7].expand(-1, k, 1).clamp(0, 1)
        opened = torch.cat([first_open, prediction[..., 3].sigmoid()], dim=2)
        return xyz, opened, geometry

    def active_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
