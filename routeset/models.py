"""Small route-set networks for the task-level planning prototype.

Every route is represented by its interior waypoint residuals relative to the
start--goal straight line.  Endpoints and reconstruction belong to the caller.
The denoiser embeds one flattened route per token; it is deliberately a small
MLP/set-attention model, not a temporal waypoint transformer.  Candidate
attention has no candidate indices or positional embeddings, so the denoiser is
permutation equivariant and can process a candidate count unseen in training.
"""

import math
from typing import Optional

import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    """Encode diffusion time in [0, 1] with fixed Fourier features."""

    def __init__(self, width: int):
        super().__init__()
        feature_count = max(8, width // 4)
        self.register_buffer("frequencies", torch.logspace(0.0, 3.0, feature_count))
        self.project = nn.Sequential(
            nn.Linear(2 * feature_count, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        phase = timesteps.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        phase = phase * (2.0 * math.pi)
        return self.project(torch.cat((phase.sin(), phase.cos()), dim=-1))


class _RouteBlock(nn.Module):
    """Conditioned token MLP, optionally followed by candidate communication."""

    def __init__(self, width: int, heads: int, communicate: bool):
        super().__init__()
        self.communicate = communicate
        self.norm = nn.LayerNorm(width)
        self.condition = nn.Linear(width, 2 * width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )
        # Always instantiate these parameters to make the architecture ablation
        # explicit. They are bypassed and receive no gradients without attention.
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(context).chunk(2, dim=-1)
        hidden = self.norm(tokens) * (1.0 + scale[:, None]) + shift[:, None]
        tokens = tokens + self.mlp(hidden)
        if self.communicate:
            hidden = self.attention_norm(tokens)
            update, _ = self.attention(hidden, hidden, hidden, need_weights=False)
            tokens = tokens + update
        return tokens

    def inactive_parameter_count(self) -> int:
        if self.communicate:
            return 0
        return sum(parameter.numel() for parameter in self.attention.parameters()) + sum(
            parameter.numel() for parameter in self.attention_norm.parameters()
        )


class RouteDenoiser(nn.Module):
    """Predict epsilon for independent or jointly denoised route candidates.

    Args:
        cond_dim: Dimension of geometry plus any externally computed VLM features.
        horizon: Total waypoint count, including the two fixed endpoints.
        set_attention: Enable permutation-equivariant candidate communication.

    ``forward(noisy, timesteps, cond)`` accepts [B,K,H-2,3], [B], [B,C].
    Timesteps are normalized to [0,1].  No route-mode labels or slot IDs enter
    this model. Identically shaped checkpoints can be used for the ablation.
    """

    def __init__(
        self,
        cond_dim: int = 12,
        horizon: int = 24,
        width: int = 192,
        depth: int = 3,
        heads: int = 4,
        set_attention: bool = True,
    ):
        super().__init__()
        if horizon < 3 or width % heads:
            raise ValueError("horizon must be >= 3 and width divisible by heads")
        self.cond_dim = cond_dim
        self.horizon = horizon
        self.set_attention = set_attention
        self.route_input = nn.Linear((horizon - 2) * 3, width)
        self.condition_encoder = nn.Sequential(
            nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.time_encoder = FourierTimeEmbedding(width)
        self.blocks = nn.ModuleList(
            [_RouteBlock(width, heads, set_attention) for _ in range(depth)]
        )
        self.route_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, (horizon - 2) * 3))

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        if noisy.ndim != 4 or tuple(noisy.shape[-2:]) != (self.horizon - 2, 3):
            raise ValueError("noisy must have shape [B,K,horizon-2,3]")
        if cond.ndim != 2 or cond.shape != (noisy.shape[0], self.cond_dim):
            raise ValueError("cond must have shape [B,cond_dim]")
        if timesteps.numel() != noisy.shape[0]:
            raise ValueError("one diffusion timestep is required per batch item")
        context = self.condition_encoder(cond) + self.time_encoder(timesteps)
        tokens = self.route_input(noisy.flatten(start_dim=2)) + context[:, None]
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.route_output(tokens).reshape_as(noisy)

    def active_parameter_count(self) -> int:
        """Count parameters that participate in this variant's forward pass."""
        total = sum(parameter.numel() for parameter in self.parameters())
        return total - sum(block.inactive_parameter_count() for block in self.blocks)


class SetRegressor(nn.Module):
    """Deterministic learned-query baseline, trained by set/Hungarian matching.

    Learned queries break symmetry among output slots. Unlike RouteDenoiser this
    is an ordered parameterization of an unordered output set; loss matching
    must be supplied by the caller. Output is straight-line waypoint residuals.
    """

    def __init__(
        self,
        cond_dim: int = 12,
        horizon: int = 24,
        max_candidates: int = 3,
        width: int = 192,
        depth: int = 3,
        heads: int = 4,
    ):
        super().__init__()
        if horizon < 3 or width % heads or max_candidates < 1:
            raise ValueError("invalid horizon, attention width, or candidate count")
        self.cond_dim = cond_dim
        self.horizon = horizon
        self.max_candidates = max_candidates
        self.queries = nn.Parameter(torch.randn(max_candidates, width) * 0.1)
        self.condition_encoder = nn.Sequential(
            nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.blocks = nn.ModuleList([_RouteBlock(width, heads, True) for _ in range(depth)])
        self.route_output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, (horizon - 2) * 3))
        # Start close to the straight line, with enough variation to give
        # matching a reproducible initial assignment and every query a gradient.
        nn.init.normal_(self.route_output[-1].weight, std=0.01)
        nn.init.zeros_(self.route_output[-1].bias)

    def forward(self, cond: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
        k = self.max_candidates if k is None else k
        if k < 1 or k > self.max_candidates:
            raise ValueError("k must be between 1 and max_candidates")
        if cond.ndim != 2 or cond.shape[-1] != self.cond_dim:
            raise ValueError("cond must have shape [B,cond_dim]")
        context = self.condition_encoder(cond)
        tokens = self.queries[:k][None].expand(cond.shape[0], -1, -1) + context[:, None]
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.route_output(tokens).reshape(cond.shape[0], k, self.horizon - 2, 3)

    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class Critic(nn.Module):
    """Score per-route task/geometry usefulness as uncalibrated logits.

    The caller supplies binary labels and a held-out calibration procedure if
    probabilities are needed. The critic does not communicate between routes,
    so its scores are invariant to the other candidates in the set.
    """

    def __init__(self, cond_dim: int = 12, horizon: int = 24, width: int = 192):
        super().__init__()
        self.cond_dim = cond_dim
        self.horizon = horizon
        self.network = nn.Sequential(
            nn.Linear((horizon - 2) * 3 + cond_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, interior: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if interior.ndim != 4 or tuple(interior.shape[-2:]) != (self.horizon - 2, 3):
            raise ValueError("interior must have shape [B,K,horizon-2,3]")
        context = cond[:, None].expand(-1, interior.shape[1], -1)
        return self.network(torch.cat((interior.flatten(start_dim=2), context), dim=-1)).squeeze(-1)
