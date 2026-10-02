"""Controlled constraint changes and matched full/local route updates.

This is a mechanism probe, not a novelty claim. Only TRAIN/DEV_MODEL parents
are accepted. Opening indices keep their physical identity when a gap closes.
Exact whole-route checks are shared by all methods. Pointwise edit labels are
training-only distance proxies, never inference inputs or collision labels.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .common import decode_paths
from .models import _RouteBlock
from .multigate import path_validity, route_modes, unpack_scene

VERSION = "multigate_change_v1"


def close_opening(scene, wall, opening):
    """Close one physical interval without shifting any other opening slot."""
    result = np.asarray(scene).copy()
    _, _, rows = unpack_scene(result)
    if wall not in (0, 1) or opening not in range(4):
        raise ValueError("invalid wall/opening")
    if rows[wall, 10 + opening] < .5 or rows[wall, 10:14].sum() < 2:
        raise ValueError("must close a present gap while leaving a traversable wall")
    result[6 + wall * 14 + 10 + opening] = 0
    return result


def checked_context(old_paths, old_scenes, new_scenes):
    """Common inference preprocessing; no reference paths or mode labels used.

    Invalid-first ordering is a controlled exact-checker rule, not a learned
    selection contribution. Stable slot order resolves ties identically.
    """
    old_valid = np.stack([path_validity(p, s)["valid"] for p, s in zip(old_paths, old_scenes)])
    new_valid = np.stack([path_validity(p, s)["valid"] for p, s in zip(old_paths, new_scenes)])
    order = np.argsort(new_valid.astype(int), axis=1, kind="stable")
    rows = np.arange(len(order))[:, None]
    return dict(drafts=old_paths[rows, order],
                valid=np.stack([old_valid[rows, order], new_valid[rows, order]], axis=-1),
                source_order=order, old_valid=old_valid, new_valid=new_valid)


def derive_changes(data, old_paths, minimum_types=6):
    """All single closures of eligible developmental layouts, no resplitting."""
    if set(data["splits"].tolist()) - {"TRAIN", "DEV_MODEL"}:
        raise ValueError("pass a physically separate TRAIN/DEV_MODEL archive")
    expected = (len(data["scenes"]), 4, data["paths"].shape[-2], 3)
    if np.shape(old_paths) != expected or not np.isfinite(old_paths).all():
        raise ValueError("four finite actual model drafts per old scene required")
    parent_splits = {}
    for parent, split in zip(data["parent_ids"], data["splits"]):
        if parent in parent_splits and parent_splits[parent] != split:
            raise ValueError("parent leaked across splits")
        parent_splits[parent] = split
    records = []
    for idx, scene in enumerate(data["scenes"]):
        if data["path_mask"][idx].sum() < minimum_types:
            continue
        _, _, walls = unpack_scene(scene)
        for wall in range(2):
            if walls[wall, 10:14].sum() < 2:
                continue
            for opening in np.flatnonzero(walls[wall, 10:14] > .5):
                changed = close_opening(scene, wall, int(opening))
                modes = data["modes"][idx].copy()
                affected = modes // 4 == opening if wall == 0 else modes % 4 == opening
                mask = data["path_mask"][idx] & ~affected
                refs = data["paths"][idx]
                if not mask.any() or not path_validity(refs[mask], changed)["valid"].all():
                    raise ValueError("a retained positive failed the unchanged checker")
                if not np.array_equal(route_modes(refs[mask], changed), modes[mask]):
                    raise ValueError("physical opening identities changed")
                records.append(dict(old_scenes=scene, scenes=changed, old_paths=old_paths[idx],
                    paths=refs, modes=modes, path_mask=mask, splits=data["splits"][idx],
                    parent_ids=data["parent_ids"][idx], source_scene_ids=data["scene_ids"][idx],
                    scene_ids="%s_close_w%d_g%d" % (data["scene_ids"][idx], wall, opening),
                    closed_wall=wall, closed_opening=opening))
    if not records:
        raise ValueError("no eligible variable-mode parents")
    result = {key: np.asarray([row[key] for row in records]) for key in records[0]}
    context = checked_context(result["old_paths"], result["old_scenes"], result["scenes"])
    result.update(old_valid=context["old_valid"], new_valid=context["new_valid"])
    return result


def training_targets(data, ids, k, threshold=.02):
    """Same source-to-positive matching and point proxy for both architectures.

    Matching is independent of learned predictions. All positive references
    participate; valid old covered types are removed, with positive fallback
    only when all known types are already covered. This is a minimum-change
    teacher protocol, not a general total-solution-count target.
    """
    ids = np.asarray(ids)
    if k not in (1, 2) or threshold < 0:
        raise ValueError("k=1/2 and nonnegative proxy threshold required")
    if np.any(data["splits"][ids] != "TRAIN"):
        raise ValueError("training-only targets must never be queried for development inference")
    context = checked_context(data["old_paths"][ids], data["old_scenes"][ids], data["scenes"][ids])
    source = context["drafts"][:, :k]
    targets, selected, remaining_counts, remaining_masks = [], [], [], []
    for row, idx in enumerate(ids):
        modes = route_modes(data["old_paths"][idx], data["scenes"][idx])
        covered = set(modes[context["new_valid"][row] & (modes >= 0)].tolist())
        mask = data["path_mask"][idx] & ~np.isin(data["modes"][idx], list(covered))
        if not mask.any():
            mask = data["path_mask"][idx].copy()
        positive_ids = np.flatnonzero(mask)
        remaining_masks.append(mask)
        remaining_counts.append(len(positive_ids))
        cost = ((source[row, :, None] - data["paths"][idx, positive_ids][None]) ** 2).mean((-1, -2))
        if len(positive_ids) < k:
            assignment = cost.argmin(1)
            ref_rows, output_cols = linear_sum_assignment((cost - cost.min(1)[:, None]).T)
            assignment[output_cols] = ref_rows
        else:
            output_rows, reference_cols = linear_sum_assignment(cost)
            assignment = np.empty(k, dtype=int)
            assignment[output_rows] = reference_cols
        chosen = positive_ids[assignment]
        selected.append(chosen)
        targets.append(data["paths"][idx, chosen])
    targets = np.stack(targets)
    proxy = np.linalg.norm(targets[:, :, 1:-1] - source[:, :, 1:-1], axis=-1) > threshold
    return dict(context=context, targets=targets, proxy=proxy, selected_reference_ids=np.stack(selected),
                remaining_positive_count=np.asarray(remaining_counts), remaining_mask=np.stack(remaining_masks))


class ConstraintUpdateRegressor(nn.Module):
    """Shared network; only the use of a predicted edit gate differs.

    Full predicts every point. Local copies unedited source points exactly;
    its hard gate uses a straight-through derivative during training. Both
    train the same gate auxiliary head on identical distance-proxy labels.
    All outputs use the same hard start/goal constraints as historical models.
    """
    def __init__(self, horizon=24, width=128, depth=2, mechanism="full"):
        super().__init__()
        if mechanism not in ("full", "local") or horizon < 3 or width % 4:
            raise ValueError("invalid model configuration")
        self.horizon, self.mechanism = horizon, mechanism
        self.queries = nn.Parameter(torch.randn(2, width) * .1)
        self.condition_encoder = nn.Sequential(nn.Linear(102, width), nn.SiLU(), nn.Linear(width, width))
        self.draft_encoder = nn.Sequential(nn.Linear(horizon * 3 + 2, width), nn.SiLU(), nn.Linear(width, width))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.blocks = nn.ModuleList([_RouteBlock(width, 4, True) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, (horizon - 2) * 3))
        self.edit_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, horizon - 2))
        nn.init.normal_(self.output[-1].weight, std=.01)
        nn.init.zeros_(self.output[-1].bias)
        nn.init.zeros_(self.edit_head[-1].weight)
        nn.init.constant_(self.edit_head[-1].bias, 2.)

    def forward(self, old_scenes, scenes, drafts, valid, k=2):
        batch = len(scenes)
        if scenes.shape != (batch, 34) or old_scenes.shape != scenes.shape:
            raise ValueError("old/new exact geometry [B,34] required")
        if drafts.shape != (batch, 4, self.horizon, 3) or valid.shape != (batch, 4, 2) or k not in (1, 2):
            raise ValueError("checked ordered drafts [B,4,H,3], validity [B,4,2], k=1/2 required")
        context = self.condition_encoder(torch.cat([old_scenes, scenes, scenes - old_scenes], dim=-1))
        route_tokens = self.draft_encoder(torch.cat([drafts.flatten(2), valid.to(drafts.dtype)], dim=-1))
        tokens = context[:, None] + route_tokens[:, :k] + self.queries[:k][None]
        read, _ = self.attention(tokens, route_tokens, route_tokens, need_weights=False)
        tokens = tokens + read
        for block in self.blocks:
            tokens = block(tokens, context)
        proposal = decode_paths(self.output(tokens).reshape(batch, k, self.horizon - 2, 3), scenes)
        logits = self.edit_head(tokens)
        hard = (logits >= 0).to(proposal.dtype)
        probability = logits.sigmoid()
        gate = probability + (hard - probability).detach() if self.training else hard
        if self.mechanism == "local":
            source = drafts[:, :k]
            # Arithmetic with zero gate copies finite coordinates bit exactly.
            interior = source[:, :, 1:-1] + gate[..., None] * (proposal[:, :, 1:-1] - source[:, :, 1:-1])
            proposal = torch.cat([proposal[:, :, :1], interior, proposal[:, :, -1:]], dim=2)
        return proposal, logits

    def active_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def update_metrics(old_paths, predictions, scenes, refs, ref_mask):
    """Budget is 4+b even if common checking later rejects a generated route."""
    from .multigate import route_metrics
    old = route_metrics(old_paths, scenes, refs, ref_mask)
    total = route_metrics(np.concatenate([old_paths, predictions], axis=1), scenes, refs, ref_mask)
    new = route_metrics(predictions, scenes, refs, ref_mask) if predictions.shape[1] else None
    rows = []
    for idx in range(len(scenes)):
        known = set(refs[idx, ref_mask[idx]].tolist())
        before = set(old["per_scene"]["modes"][idx].tolist()) - {-1}
        after = set(total["per_scene"]["modes"][idx].tolist()) - {-1}
        rows.append(dict(total_candidates=4 + predictions.shape[1], new_candidates=predictions.shape[1],
            old_invalid_count=int(4 - old["per_scene"]["valid"][idx].sum()),
            old_unique_valid=len(before), additional_unique_valid=len(after - before),
            union_unique_valid=len(after), union_reference_coverage=len(after & known) / len(known),
            union_valid_rate=float(total["per_scene"]["valid"][idx].mean()),
            union_any_valid=bool(total["per_scene"]["valid"][idx].any()),
            new_valid_rate=float(new["per_scene"]["valid"][idx].mean()) if new else None,
            new_collision_rate=float(new["per_scene"]["collision"][idx].mean()) if new else None,
            retained_old_valid_count=int(old["per_scene"]["valid"][idx].sum()),
            reference_types=len(known)))
    return rows


def aggregate_parents(rows):
    """Changes are not independent scenes; average each parent, then parents."""
    if not rows:
        raise ValueError("nonempty rows required")
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float, bool))]
    parents = sorted(set(row["parent_id"] for row in rows))
    result = {key: float(np.mean([np.mean([r[key] for r in rows if r["parent_id"] == parent])
                                for parent in parents])) for key in keys}
    result.update(parent_count=len(parents), change_count=len(rows), aggregation="equal-parent mean over all eligible closures")
    return result
