"""Training-only exact signed L-infinity segment/AABB clearance.

This matches the existing checker's axis-expanded boxes, not Euclidean SDF.
The minimum of max(abs(p(t)-center)-halfsize) is attained at an endpoint
or an intersection of two of its six affine faces. Negative values retain
penetration gradients. Piecewise differentiable; no sampled collision claim.
"""
import torch


def segment_box_signed_clearance(p0, p1, center, halfsize):
    """Broadcast [...,3] inputs; return exact segment minimum per box."""
    offset = p0-center
    direction = p1-p0
    intercept = torch.cat((offset-halfsize, -offset-halfsize), -1)
    slope = torch.cat((direction, -direction), -1)
    slope, intercept = torch.broadcast_tensors(slope, intercept)
    i, j = torch.triu_indices(6, 6, 1, device=p0.device)
    denominator = slope[..., i]-slope[..., j]
    moving = denominator.abs() > 1e-12
    safe = torch.where(moving, denominator, torch.ones_like(denominator))
    t = ((intercept[..., j]-intercept[..., i])/safe).clamp(0, 1)
    t = torch.where(moving, t, torch.zeros_like(t))
    t = torch.cat((torch.zeros_like(t[..., :1]), torch.ones_like(t[..., :1]), t), -1)
    values = intercept[..., None, :]+t[..., :, None]*slope[..., None, :]
    return values.max(-1).values.min(-1).values


def path_segment_clearances(paths, centers, halfsizes):
    return segment_box_signed_clearance(
        paths[:, :, :-1, None, :], paths[:, :, 1:, None, :],
        centers[:, None, None], halfsizes[:, None, None]).min(-1).values


def segment_clearance_loss(paths, centers, halfsizes, margin=.02):
    """Equal weight all B*M*(H-1) segments; worst physical box per segment."""
    clearance = path_segment_clearances(paths, centers, halfsizes)
    return (margin-clearance).clamp_min(0).square().mean()
