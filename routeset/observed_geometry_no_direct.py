"""Ordinary reduced-capacity control; geometry retains its Qwen conditioning."""
import hashlib

import torch
from torch import nn

from .observed_geometry import ObservedGeometryRouteHead
from .observed_training_audit import tensor_state_digest

PROTOCOL = 'ordinary_geometry_no_direct_decoder_qwen_v1'
REMOVED_PREFIX = 'head.feature_encoder.'


class _ZeroDirectFeature(nn.Module):
    """Parameter-free zero summand; no input values or labels are inspected."""
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, features):
        return features.new_zeros((len(features), self.width))


class ObservedGeometryNoDirectHead(ObservedGeometryRouteHead):
    """Construct in the historical order, then remove only the direct encoder.

    Inheriting the original forward preserves all geometry/endpoint/event
    equations. Its direct summand is exactly zero. This does NOT remove the
    language query within geometry.fusion, freeze geometry, or clamp endpoints.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.refiner is not None or self.endpoint_mode != 'surface_anchor':
            raise ValueError('No-direct control requires surface anchor and no refiner')
        before = self.state_dict()
        full_hash = tensor_state_digest(before)
        shared = {k: v for k, v in before.items() if not k.startswith(REMOVED_PREFIX)}
        shared_hash = tensor_state_digest(shared)
        removed_count = sum(p.numel() for p in self.head.feature_encoder.parameters())
        original_count = self.active_parameter_count()
        rng = torch.get_rng_state().clone()
        width = self.head.queries.shape[-1]
        self.head.feature_encoder = _ZeroDirectFeature(width)
        if not torch.equal(rng, torch.get_rng_state()) or tensor_state_digest(self.state_dict()) != shared_hash:
            raise RuntimeError('Removing direct branch changed shared initialization or RNG')
        self.initialization_receipt = dict(protocol=PROTOCOL, original_model_sha256=full_hash,
            shared_initialization_sha256=shared_hash,
            post_construction_torch_rng_sha256=hashlib.sha256(rng.numpy().tobytes()).hexdigest(),
            original_parameters=original_count, removed_parameters=removed_count,
            remaining_parameters=self.active_parameter_count(),
            removed_parameter_names=sorted(k for k in before if k.startswith(REMOVED_PREFIX)),
            all_remaining_parameters_trainable=all(p.requires_grad for p in self.parameters()),
            shared_tensor_bytes_exact=True, rng_exact=True, initial_forward_equal_claim=False,
            initial_parameter_sha256={k: tensor_state_digest({k: v}) for k, v in self.state_dict().items()})
