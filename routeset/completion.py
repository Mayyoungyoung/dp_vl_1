"""Paired route completion: plain attention versus an idempotent coverage pool.

This is a controlled-geometry mechanism probe, not a novelty claim. Both
variants receive exactly the same route coordinates and exact-validity gate.
Mode IDs are used only to construct positive training targets and evaluation.
With at most two drafts, [A,A] is invariant for attention too; that case checks
correctness and does not establish an advantage of max pooling. The paired
experiment primarily tests whether distinct covered-route features combine
more usefully under max than under attention averaging.
"""

import math

import numpy as np
import torch
from torch import nn

from .models import _RouteBlock
from .multigate import path_validity, route_modes, wall_boxes


CONTEXT_KINDS = ("empty", "single", "two_valid", "duplicate", "invalid", "mixed")
CONTEXT_COUNTS = dict(empty=0, single=1, two_valid=2, duplicate=2, invalid=2, mixed=2)


class CompletionRegressor(nn.Module):
    """Same parameters in both variants; only the parameter-free pool differs."""

    def __init__(self, cond_dim=34, horizon=24, width=128, depth=2,
                 mechanism="attention", max_candidates=4):
        super().__init__()
        if mechanism not in ("attention", "coverage") or width % 4 or horizon < 3:
            raise ValueError("invalid mechanism, width or horizon")
        self.mechanism, self.horizon = mechanism, horizon
        self.cond_dim, self.max_candidates = cond_dim, max_candidates
        self.queries = nn.Parameter(torch.randn(max_candidates, width) * .1)
        self.condition_encoder = nn.Sequential(nn.Linear(cond_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.draft_encoder = nn.Sequential(nn.Linear(horizon * 3, width), nn.SiLU(), nn.Linear(width, width))
        self.context_projection = nn.Linear(width, width, bias=False)
        self.blocks = nn.ModuleList([_RouteBlock(width, 4, True) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, (horizon - 2) * 3))
        nn.init.normal_(self.output[-1].weight, std=.01)
        nn.init.zeros_(self.output[-1].bias)

    def draft_summary(self, tokens, drafts, draft_valid):
        # Invalid drafts cannot leak either their coordinates or encoder bias.
        clean = torch.where(draft_valid[..., None, None], drafts, torch.zeros_like(drafts))
        encoded = self.draft_encoder(clean.flatten(start_dim=2)).sigmoid()
        if self.mechanism == "coverage":
            pooled = (encoded * draft_valid[..., None]).amax(dim=1)
            return pooled[:, None].expand(-1, tokens.shape[1], -1)
        logits = torch.einsum("bkw,bdw->bkd", tokens, encoded) / math.sqrt(tokens.shape[-1])
        # A finite sentinel and post-softmax masking avoid NaNs for empty sets.
        weights = logits.masked_fill(~draft_valid[:, None], -1e4).softmax(dim=-1)
        weights = weights * draft_valid[:, None]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("bkd,bdw->bkw", weights, encoded)

    def forward(self, scenes, drafts, draft_valid, k=2):
        if scenes.ndim != 2 or scenes.shape[-1] != self.cond_dim:
            raise ValueError("scenes must be [B,cond_dim]")
        if drafts.shape != (len(scenes), 2, self.horizon, 3) or draft_valid.shape != (len(scenes), 2):
            raise ValueError("drafts [B,2,H,3] and draft_valid [B,2] required")
        if not 1 <= k <= self.max_candidates:
            raise ValueError("candidate budget outside model capacity")
        draft_valid = draft_valid.bool()
        context = self.condition_encoder(scenes)
        tokens = context[:, None] + self.queries[:k][None]
        tokens = tokens + self.context_projection(self.draft_summary(tokens, drafts, draft_valid))
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.output(tokens).reshape(len(scenes), k, self.horizon - 2, 3)

    def active_parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def draft_validity(drafts, scenes, present):
    """CPU exact full-segment validity, identical and timed for both methods."""
    return np.stack([path_validity(paths, scene)["valid"] for paths, scene in zip(drafts, scenes)]) & present


def build_contexts(data, ids, rng, kinds=None):
    """Return raw-coordinate contexts and remaining positive reference masks.

    Invalid contexts are deliberately placed through an obstacle then checked,
    rather than treating arbitrary perturbations as automatically invalid.
    """
    ids = np.asarray(ids)
    if kinds is None:
        kinds = rng.choice(CONTEXT_KINDS, size=len(ids))
    if isinstance(kinds, str):
        kinds = [kinds] * len(ids)
    if len(kinds) != len(ids):
        raise ValueError("one context kind required per scene")
    horizon = data["paths"].shape[-2]
    drafts = np.zeros((len(ids), 2, horizon, 3), dtype=np.float32)
    present = np.zeros((len(ids), 2), dtype=bool)
    for row, (idx, kind) in enumerate(zip(ids, kinds)):
        if kind not in CONTEXT_KINDS:
            raise ValueError("unknown context kind")
        positives = np.flatnonzero(data["path_mask"][idx])
        if not len(positives):
            raise ValueError("completion training requires known positive references")
        choices = rng.choice(positives, size=2, replace=len(positives) < 2)
        count = CONTEXT_COUNTS[kind]
        present[row, :count] = True
        if count:
            drafts[row, :count] = data["paths"][idx, choices[:count]]
        if kind == "duplicate":
            drafts[row, 1] = drafts[row, 0]
        if kind in ("invalid", "mixed"):
            lower, upper = wall_boxes(data["scenes"][idx])
            obstacle = (lower[0] + upper[0]) / 2
            invalid_slots = (0, 1) if kind == "invalid" else (1,)
            for slot in invalid_slots:
                drafts[row, slot, horizon // 2] = obstacle
    valid = draft_validity(drafts, data["scenes"][ids], present)
    remaining = data["path_mask"][ids].copy()
    covered = np.zeros(len(ids), dtype=np.int64)
    fallback = np.zeros(len(ids), dtype=bool)
    for row, idx in enumerate(ids):
        labels = route_modes(drafts[row], data["scenes"][idx])
        known_covered = set(labels[valid[row] & (labels >= 0)].tolist())
        covered[row] = len(known_covered)
        for label in known_covered:
            remaining[row] &= data["modes"][idx] != label
        if not remaining[row].any():
            remaining[row] = data["path_mask"][idx]
            fallback[row] = True
    return dict(drafts=drafts, present=present, valid=valid, remaining_mask=remaining,
                kinds=np.asarray(kinds), covered_count=covered, all_known_covered=fallback)
