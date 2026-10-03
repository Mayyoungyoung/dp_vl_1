"""Shared representations and strict scene-level data partitions."""
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path, features=None):
    with np.load(path, allow_pickle=False) as f:
        data = {key: f[key] for key in f.files}
    condition = data['scenes'].astype(np.float32)
    if features:
        with np.load(features, allow_pickle=False) as f:
            assert np.array_equal(f['scene_ids'], data['scene_ids']), 'feature/data scene mismatch'
            expected = str(f['dataset_sha256'].item())
            assert expected == sha256(path), 'feature cache belongs to another dataset'
            extra = f['features'].astype(np.float32)
        condition = np.concatenate([condition, extra], axis=-1)
    data['condition'] = condition
    return data


def straight_line(scenes, horizon):
    import torch
    t = torch.linspace(0, 1, horizon, device=scenes.device, dtype=scenes.dtype)
    return scenes[:, None, :3] * (1-t[None, :, None]) + scenes[:, None, 3:6] * t[None, :, None]


def encode_paths(paths, scenes):
    # Endpoints are hard constraints shared by every method.
    base = straight_line(scenes, paths.shape[-2])
    return paths[..., 1:-1, :] - base[:, None, 1:-1, :]


def decode_paths(residual, scenes):
    base = straight_line(scenes, residual.shape[-2] + 2)
    paths = base[:, None].expand(-1, residual.shape[1], -1, -1).clone()
    paths[:, :, 1:-1] += residual
    return paths


def balanced_targets(data, indices, k, rng):
    result = []
    for idx in indices:
        mode_ids = np.unique(data['modes'][idx])
        chosen = []
        for mode in rng.permutation(mode_ids)[:min(k, len(mode_ids))]:
            choices = np.flatnonzero(data['modes'][idx] == mode)
            chosen.append(rng.choice(choices))
        while len(chosen) < k:
            chosen.append(rng.integers(data['paths'].shape[1]))
        result.append(data['paths'][idx, rng.permutation(chosen)])
    return np.stack(result).astype(np.float32)


def matching_loss(pred, target):
    import torch
    # Exhaustive Hungarian-equivalent assignment for tiny candidate budgets.
    k = pred.shape[1]
    if k > 6:
        raise ValueError('Exhaustive matching supports at most six candidates')
    losses = []
    for p in itertools.permutations(range(k)):
        losses.append((pred[:, p] - target).square().mean(dim=(1, 2, 3)))
    return torch.stack(losses, dim=1).min(dim=1).values.mean()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
