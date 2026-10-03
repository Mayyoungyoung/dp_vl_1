"""Frozen-Qwen observation-only route baseline with learned 3-D endpoints.

Reference routes and semantic target coordinates are supervision/evaluation
fields only. They never enter the condition encoder. This pilot cannot certify
continuous collision validity, route types or complete robot execution.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import sha256
from .models import _RouteBlock


QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
OBSERVATION_KEYS = {"id", "parent_id", "split", "image", "instruction"}
CACHE_KEYS = {"mean_hidden", "last_hidden", "id", "parent_id", "split", "image_sha256", "input_tokens"}
OBSERVATION_EVAL_PROTOCOL = "observation_eval_v2"


def resample_event_segments(poses, gripper_open, horizon=24):
    """Resample each constant-gripper-state phase separately, keeping boundaries.

    Each nontrivial phase retains its first and last recorded point. Point
    budgets beyond this minimum follow phase arc length. A too-small horizon
    raises an error instead of erasing grasp/release transitions.
    """
    xyz = np.asarray(poses, dtype=np.float64)[:, :3]
    states = np.asarray(gripper_open).reshape(-1) > .5
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(states) or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("finite [T,>=3] poses and matching gripper states required")
    cuts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    phases = [xyz[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
    minima = np.array([min(2, len(phase)) for phase in phases])
    if minima.sum() > horizon:
        raise ValueError("horizon cannot preserve all event phase boundaries")
    lengths = np.array([np.linalg.norm(np.diff(phase, axis=0), axis=1).sum() for phase in phases])
    weights = lengths if lengths.sum() > 1e-10 else np.ones(len(phases))
    quotas = (horizon - minima.sum()) * weights / weights.sum()
    allocations = minima + np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - np.floor(quotas)), kind="stable")[:horizon - allocations.sum()]:
        allocations[index] += 1
    output, events = [], []
    for index, (phase, count) in enumerate(zip(phases, allocations)):
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(phase, axis=0), axis=1))]
        if distance[-1] < 1e-10:
            sampled = np.repeat(phase[:1], count, axis=0)
        else:
            samples = np.linspace(0., distance[-1], count)
            sampled = np.stack([np.interp(samples, distance, phase[:, axis]) for axis in range(3)], axis=1)
            sampled[0], sampled[-1] = phase[0], phase[-1]
        output.append(sampled)
        events.append(np.full(count, states[cuts[index]], dtype=np.float32))
    return np.concatenate(output).astype(np.float32), np.concatenate(events)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_observed_dataset(observations, supervision, cache_dir, horizon=24, pooling="both"):
    """Join strict observation-only Qwen caches to separately stored labels."""
    observations, supervision, cache_dir = Path(observations), Path(supervision), Path(cache_dir)
    rows, label_rows = _jsonl(observations), _jsonl(supervision)
    labels = {row["id"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("duplicate supervision id")
    config_path = cache_dir / "cache_config.json"
    cache_config = json.loads(config_path.read_text(encoding="utf-8"))
    if cache_config.get("model") != "Qwen/Qwen3-VL-2B-Instruct" or cache_config.get("revision") != QWEN_REVISION:
        raise ValueError("real pinned Qwen3-VL-2B cache required")
    if cache_config.get("model_trainable_parameter_count") != 0 or cache_config.get("manifest_sha256") != sha256(observations):
        raise ValueError("cache must be frozen and match this exact observation manifest")
    if set(cache_config.get("input_contract", [])) != OBSERVATION_KEYS:
        raise ValueError("cache input contract includes unapproved information")
    if pooling not in ("mean", "last", "both"):
        raise ValueError("pooling must be mean, last or both")
    source_hashes = {str(observations): sha256(observations), str(supervision): sha256(supervision), str(config_path): sha256(config_path)}
    samples, skipped, unreferenced, seen, parent_splits = [], [], [], set(), {}
    for row in rows:
        if set(row) != OBSERVATION_KEYS or row["id"] in seen:
            raise ValueError("observation manifest must have unique ids and exactly observation-only fields")
        seen.add(row["id"])
        parent = row["parent_id"]
        if parent in parent_splits and parent_splits[parent] != row["split"]:
            raise ValueError("parent scene leaked across splits")
        parent_splits[parent] = row["split"]
        label = labels.get(row["id"])
        if label is None:
            skipped.append(dict(id=row["id"], reason="missing_supervision"))
            continue
        if label["parent_id"] != parent or label.get("split", row["split"]) != row["split"]:
            raise ValueError("supervision identity/split mismatch")
        if not label.get("routes"):
            # A failed collector does not remove this observation from semantic
            # evaluation, and is never a label that no valid route exists.
            unreferenced.append(dict(id=row["id"], split=row["split"], reason="no_successful_reference_routes"))
        row_key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
        cache_path = cache_dir / (row_key + ".npz")
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != CACHE_KEYS:
                raise ValueError("unapproved cached feature fields")
            for key in ("id", "parent_id", "split"):
                if str(archive[key].item()) != row[key]:
                    raise ValueError("cached observation identity mismatch")
            image_path = _resolve(observations.parent, row["image"])
            image_hash = sha256(image_path)
            if str(archive["image_sha256"].item()) != image_hash:
                raise ValueError("cached image hash mismatch")
            fields = ["mean_hidden", "last_hidden"] if pooling == "both" else [pooling + "_hidden"]
            feature = np.concatenate([archive[field].reshape(-1) for field in fields]).astype(np.float32)
        current_path = _resolve(supervision.parent, label["observation"])
        with np.load(current_path, allow_pickle=False) as observation:
            # A strict whitelist prevents target/geometry/future fields from
            # accidentally entering a convenient flattened observation vector.
            current = np.r_[observation["gripper_pose"].reshape(7), observation["gripper_open"].reshape(1)].astype(np.float32)
        paths, events = [], []
        for filename in label.get("routes", []):
            route_path = _resolve(supervision.parent, filename)
            with np.load(route_path, allow_pickle=False) as archive:
                path, event = resample_event_segments(archive["gripper_pose"], archive["gripper_open"], horizon)
            if np.linalg.norm(path[0] - current[:3]) > .005:
                raise ValueError("reference does not start at the recorded current end effector")
            paths.append(path)
            events.append(event)
            source_hashes[str(route_path)] = sha256(route_path)
        if not np.isfinite(feature).all() or not np.isfinite(current).all():
            raise ValueError("non-finite model input")
        source_hashes[str(cache_path)], source_hashes[str(current_path)] = sha256(cache_path), sha256(current_path)
        samples.append(dict(id=row["id"], parent_id=parent, split=row["split"], feature=feature, current=current,
                            paths=paths, events=events, image_hash=image_hash, instruction=row["instruction"],
                            semantic_targets=label.get("semantic_targets"), task=label.get("task")))
    if not samples:
        raise ValueError("no supervised examples with cached real-Qwen observations")
    reference_count = max(len(sample["paths"]) for sample in samples)
    paths = np.zeros((len(samples), reference_count, horizon, 3), dtype=np.float32)
    events = np.zeros((len(samples), reference_count, horizon), dtype=np.float32)
    mask = np.zeros((len(samples), reference_count), dtype=bool)
    for row, sample in enumerate(samples):
        count = len(sample["paths"])
        if count:
            paths[row, :count], events[row, :count], mask[row, :count] = sample["paths"], sample["events"], True
    fingerprint = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    return dict(features=np.stack([s["feature"] for s in samples]), current=np.stack([s["current"] for s in samples]),
                paths=paths, events=events, path_mask=mask, splits=np.asarray([s["split"] for s in samples]),
                scene_ids=np.asarray([s["id"] for s in samples]), parent_ids=np.asarray([s["parent_id"] for s in samples]),
                semantic_targets=[s["semantic_targets"] for s in samples], image_hashes=np.asarray([s["image_hash"] for s in samples]),
                instructions=[s["instruction"] for s in samples], skipped=skipped, source_hashes=source_hashes,
                tasks=[s['task'] for s in samples],
                fingerprint=fingerprint, cache_config=cache_config, unreferenced=unreferenced,
                evaluation_protocol=OBSERVATION_EVAL_PROTOCOL)


class ObservedRouteHead(nn.Module):
    """Trainable set head; only the first xyz/open state is a hard constraint."""

    def __init__(self, feature_dim, horizon=24, max_candidates=4, width=128, depth=2):
        super().__init__()
        self.horizon, self.max_candidates = horizon, max_candidates
        self.feature_encoder = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.state_encoder = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(max_candidates, width) * .1)
        self.blocks = nn.ModuleList([_RouteBlock(width, 4, True) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, (horizon - 1) * 4))
        nn.init.normal_(self.output[-1].weight, std=.01)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features, current, k=None):
        k = self.max_candidates if k is None else k
        if current.shape != (len(features), 8) or not 1 <= k <= self.max_candidates:
            raise ValueError("current [B,8] and valid candidate budget required")
        context = self.feature_encoder(features) + self.state_encoder(current)
        tokens = context[:, None] + self.queries[:k][None]
        for block in self.blocks:
            tokens = block(tokens, context)
        prediction = self.output(tokens).reshape(len(features), k, self.horizon - 1, 4)
        xyz = current[:, None, None, :3] + prediction[..., :3]
        xyz = torch.cat([current[:, None, None, :3].expand(-1, k, 1, -1), xyz], dim=2)
        first_open = current[:, None, None, 7].expand(-1, k, 1).clamp(0, 1)
        opened = torch.cat([first_open, prediction[..., 3].sigmoid()], dim=2)
        return xyz, opened

    def active_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def semantic_endpoint_accuracy(endpoints, specification):
    """Evaluation-only nearest-target identity + explicit spatial tolerance."""
    if specification is None:
        return None
    centers = np.asarray(specification["centers"], dtype=np.float64)
    expected = int(specification["target_index"])
    tolerance = float(specification["tolerance"])
    if centers.ndim != 2 or centers.shape[1] != 3 or not 0 <= expected < len(centers) or tolerance <= 0:
        raise ValueError("invalid evaluation-only semantic target specification")
    distances = np.linalg.norm(np.asarray(endpoints)[:, None] - centers[None], axis=-1)
    return (distances.argmin(axis=1) == expected) & (distances[:, expected] <= tolerance)
