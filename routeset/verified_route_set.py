"""Training-only verified targets. No oracle is used by the learned forward.

Finite-reference matching is an existing control, not a new algorithm. Regions
below certify obstacle/floor clearance only; selected targets must additionally
pass the independent task/event/mode checker supplied by the caller.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from routeset.geometry import segment_aabb_intersection


def signed_clearances(paths, centers, halves):
    """Exact signed L-infinity segment/AABB clearance, NumPy implementation."""
    p = np.asarray(paths, dtype=np.float64)
    a = p[:, :-1, None] - np.asarray(centers)[None, None]
    v = (p[:, 1:] - p[:, :-1])[:, :, None]
    intercept = np.concatenate((a-halves, -a-halves), -1)
    slope = np.broadcast_to(np.concatenate((v, -v), -1), intercept.shape)
    i, j = np.triu_indices(6, 1)
    den = slope[..., i]-slope[..., j]
    moving = np.abs(den) > 1e-12
    t = np.zeros_like(den)
    np.divide(intercept[..., j]-intercept[..., i], den, out=t, where=moving)
    t = np.concatenate((np.zeros_like(t[..., :1]), np.ones_like(t[..., :1]),
                        np.clip(t, 0, 1)), -1)
    return (intercept[..., None, :] + t[..., :, None]*slope[..., None, :]).max(-1).min(-1).min(-1)


def certified_radii(paths, centers, halves, floor, margin=.02, cap=.02):
    """Independent coordinate boxes. Strict half-slack protects whole segments.

Endpoints are fixed. A zero radius is allowed at the workspace floor. There is
no certification of semantic relations, modes, IK, or whole-arm execution.
"""
    p = np.asarray(paths, dtype=np.float64)
    slack = signed_clearances(p, centers, np.asarray(halves)+margin)
    if not np.isfinite(p).all() or (slack <= 0).any() or (p[..., 2] < floor).any():
        raise ValueError('Region witnesses must be continuously clear and above floor')
    radius = np.full(p.shape[:2], cap, dtype=np.float64)
    radius[:, :-1] = np.minimum(radius[:, :-1], slack*.5)
    radius[:, 1:] = np.minimum(radius[:, 1:], slack*.5)
    radius = np.minimum(radius, (p[..., 2]-floor)*.5)
    radius[:, [0, -1]] = 0
    return radius.astype(np.float32)


def verify_reach(paths, events, current, goals, target_index, centers, halves, floor):
    """Vectorized training verifier for the unchanged V3 reach contract.

Must be replay-audited against check_candidates before a training launch.
"""
    p = np.asarray(paths, dtype=np.float64)
    e = np.asarray(events)
    if p.ndim != 3 or p.shape[-1] != 3 or e.shape != p.shape[:2]:
        raise ValueError('Expected paths[K,H,3] and events[K,H]')
    finite = np.isfinite(p).all((1, 2)) & np.isfinite(e).all(1)
    distances = np.linalg.norm(p[:, -1, None]-np.asarray(goals)[None], axis=-1)
    goal = (distances.argmin(-1) == target_index) & (distances[:, target_index] <= .03)
    start = np.linalg.norm(p[:, 0]-np.asarray(current['gripper_pose'])[:3], axis=-1) <= .005
    event_ok = ((e > .5) == bool(np.asarray(current['gripper_open']) > .5)).all(1)
    shape = (len(p), p.shape[1]-1, len(centers), 3)
    a = np.broadcast_to(p[:, :-1, None], shape)
    b = np.broadcast_to(p[:, 1:, None], shape)
    hit = segment_aabb_intersection(a, b, np.asarray(centers)-halves-.02,
                                   np.asarray(centers)+halves+.02)
    return finite & goal & start & event_ok & ~hit.any((1, 2)) & (p[..., 2].min(1) >= floor)


def representatives(valid, words):
    """Stable first-slot representative; unknown signatures are not invented."""
    protected = np.zeros(len(valid), dtype=bool)
    seen = set()
    for k, (ok, word) in enumerate(zip(valid, words)):
        if ok and word is not None and word not in seen:
            protected[k] = True
            seen.add(word)
    return protected, seen


def build_targets(paths, events, references, reference_events, tags, radii,
                  valid, words, arm, rng, check):
    """Build detached targets for one request, with explicit budget handling.

ordinary: existing nearest-within-mode injective assignment.
gate: same assignment, then stop regression of valid distinct predictions.
project: protect representatives, assign free slots to uncovered modes, then
         nearest coordinate-box projection (with independent revalidation).
No global nearest-feasible-solution or no-forgetting guarantee is made.
"""
    if arm not in ('ordinary', 'gate', 'project'):
        raise ValueError('Unknown arm')
    p, e = np.asarray(paths), np.asarray(events)
    r, re = np.asarray(references), np.asarray(reference_events)
    tags = np.asarray(tags)
    if not np.isfinite(p).all() or not np.isfinite(e).all():
        raise ValueError('Nonfinite training prediction')
    if len(r) == 0 or len(r) != len(tags) or re.shape != r.shape[:2]:
        raise ValueError('Nonempty aligned references required')
    protected, seen = representatives(valid, words)
    # Consume identical group RNG regardless of arm or predicted validity.
    groups = sorted(set(tags.tolist()))
    order = rng.permutation(len(groups))
    ordered = [groups[j] for j in order]
    union_count = len(set(groups) | seen)
    candidates = np.broadcast_to(r[None], (len(p),)+r.shape).copy()
    if arm == 'project':
        radius = np.asarray(radii)[None, :, :, None]
        candidates = np.clip(p[:, None], r[None]-radius, r[None]+radius)
    event_candidates = np.broadcast_to(re[None], (len(p),)+re.shape)
    costs = (np.square(p[:, None, 1:]-candidates[:, :, 1:]).sum((2, 3))
             + .04*np.square(e[:, None, 1:]-event_candidates[:, :, 1:]).sum(2))/(4*(p.shape[1]-1))
    choice = costs.argmin(1)
    free = np.flatnonzero(~protected) if arm == 'project' else np.arange(len(p))
    missing = [g for g in ordered if g not in seen] if arm == 'project' else ordered
    selected_groups = missing[:len(free)]
    if selected_groups:
        group_indices = [np.flatnonzero(tags == g) for g in selected_groups]
        group_choices = np.stack([idx[costs[:, idx].argmin(1)] for idx in group_indices], 1)
        group_cost = costs[np.arange(len(p))[:, None], group_choices]
        fit = costs[np.arange(len(p)), choice]
        gr, slots = linear_sum_assignment((group_cost[free]-fit[free, None]).T)
        choice[free[slots]] = group_choices[free[slots], gr]
    result = candidates[np.arange(len(p)), choice].copy()
    result_e = re[choice].copy()
    assigned = tags[choice].astype(object)
    if arm in ('gate', 'project'):
        result[protected], result_e[protected] = p[protected], e[protected]
        assigned[protected] = np.asarray(words, dtype=object)[protected]
    checked, checked_words = check(result, result_e)
    fallback = np.zeros(len(p), dtype=bool)
    if arm == 'project':
        fallback = (~checked | (np.asarray(checked_words, dtype=object) != assigned)) & ~protected
        result[fallback], result_e[fallback] = r[choice[fallback]], re[choice[fallback]]
        if fallback.any():
            checked, checked_words = check(result, result_e)
    if not np.all(checked):
        raise ValueError('Unverified supervision target')
    return result.astype(p.dtype), result_e.astype(e.dtype), dict(
        protected=protected, fallback=fallback, reference_index=choice,
        target_words=checked_words, oversubscribed=union_count > len(p),
        known_modes=len(groups), union_modes=union_count,
        unmatched_modes=max(0, len(missing)-len(free)))
