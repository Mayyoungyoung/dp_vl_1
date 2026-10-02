"""Development-only sampling and a fixed Particle-Guidance-inspired adapter.

Historical models/schedules are deliberately unchanged. Mode labels balance
TRAIN positives only; they are never model inputs or inference guidance.
"""
import hashlib

import numpy as np
import torch
from torch import nn

from .multigate import load_dataset


def load_development(path):
    data = load_dataset(path)
    if set(data["splits"].tolist()) != {"TRAIN", "DEV_MODEL"}:
        raise ValueError("nonempty TRAIN/DEV_MODEL-only archive required; locked splits forbidden")
    train = np.flatnonzero(data["splits"] == "TRAIN")
    dev = np.flatnonzero(data["splits"] == "DEV_MODEL")
    if set(data["parent_ids"][train]) & set(data["parent_ids"][dev]):
        raise ValueError("parent split leakage")
    valid = data["path_mask"].astype(bool) & (data["modes"] >= 0)
    if not valid.any(axis=1).all():
        raise ValueError("every scene requires at least one positive reference")
    if not np.isfinite(data["scenes"]).all() or not np.isfinite(data["paths"][valid]).all():
        raise ValueError("nonfinite condition or positive reference")
    return data, train, dev


class PairedTrainingStream:
    """Independent parent, positive-target and CPU diffusion-noise RNG streams.

    Their serialized state and rolling byte digest verify actual matching even
    when the models consume different random numbers. The digest includes all
    selected scene/target indices, timesteps and Gaussian noise, not just seeds.
    """
    def __init__(self, data, train_ids, seed):
        self.data = data
        self.groups = [train_ids[data["parent_ids"][train_ids] == parent]
                       for parent in np.unique(data["parent_ids"][train_ids])]
        self.parents = np.random.default_rng(seed + 100000)
        self.targets = np.random.default_rng(seed + 200000)
        self.noise = torch.Generator(device="cpu").manual_seed(seed + 300000)
        self.digest = "0" * 64

    def draw(self, batch_size, k, diffusion_steps):
        draws = self.parents.integers(len(self.groups), size=batch_size)
        ids = np.asarray([self.parents.choice(self.groups[p]) for p in draws], dtype=np.int64)
        selected = []
        for idx in ids:
            if self.data["splits"][idx] != "TRAIN":
                raise ValueError("training references must come from TRAIN")
            mask = self.data["path_mask"][idx] & (self.data["modes"][idx] >= 0)
            modes = np.unique(self.data["modes"][idx, mask])
            choices = []
            for mode in self.targets.permutation(modes)[:k]:
                choices.append(self.targets.choice(np.flatnonzero(mask & (self.data["modes"][idx] == mode))))
            while len(choices) < k:
                # Repeated valid modes are allowed when fewer than K are known.
                mode = self.targets.choice(modes)
                choices.append(self.targets.choice(np.flatnonzero(mask & (self.data["modes"][idx] == mode))))
            selected.append(self.targets.permutation(choices))
        selected = np.asarray(selected, dtype=np.int64)
        paths = self.data["paths"][ids[:, None], selected].astype(np.float32)
        times = torch.randint(diffusion_steps, (batch_size,), generator=self.noise)
        noise = torch.randn(batch_size, k, paths.shape[-2] - 2, 3, generator=self.noise)
        digest = hashlib.sha256(bytes.fromhex(self.digest))
        for array in (ids, selected, times.numpy(), noise.numpy()):
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
        self.digest = digest.hexdigest()
        return ids, selected, paths, times, noise

    def state_dict(self):
        return dict(parents=self.parents.bit_generator.state, targets=self.targets.bit_generator.state,
                    noise=self.noise.get_state(), digest=self.digest)

    def load_state_dict(self, state):
        self.parents.bit_generator.state = state["parents"]
        self.targets.bit_generator.state = state["targets"]
        self.noise.set_state(state["noise"].cpu())
        self.digest = state["digest"]


def rbf_repulsion(noisy, alpha_bar, bandwidth=.2):
    """Analytic -grad E for the documented normalized, symmetric RBF potential."""
    if bandwidth <= 0:
        raise ValueError("positive bandwidth required")
    if noisy.shape[1] == 1:
        return torch.zeros_like(noisy)
    flat = noisy.flatten(start_dim=2)
    delta = flat[:, :, None] - flat[:, None, :]
    alpha = torch.as_tensor(alpha_bar, device=noisy.device, dtype=noisy.dtype).reshape(-1, 1, 1)
    scale = flat.shape[-1] * (alpha * bandwidth ** 2 + 1 - alpha)
    kernel = torch.exp(-delta.square().sum(-1) / (2 * scale))
    force = (kernel[..., None] * delta / scale[..., None]).sum(2) / (flat.shape[1] - 1)
    return force.reshape_as(noisy)


class _GuidedDenoiser(nn.Module):
    def __init__(self, model, schedule, strength, bandwidth):
        super().__init__()
        self.model, self.schedule = model, schedule
        self.strength, self.bandwidth = strength, bandwidth
        self.horizon = model.horizon

    def forward(self, noisy, normalized_times, condition):
        epsilon = self.model(noisy, normalized_times, condition)
        times = (normalized_times * (self.schedule.num_steps - 1)).round().long()
        alpha = self.schedule.alpha_bars[times].to(noisy)
        force = rbf_repulsion(noisy, alpha, self.bandwidth)
        return epsilon - self.strength * (1 - alpha).sqrt()[:, None, None, None] * force


def guided_sample(schedule, model, condition, k, steps=40, generator=None, strength=0., bandwidth=.2):
    """Exactly K particles and the original number of denoiser evaluations.

    Fixed-potential adaptation, not an official full reproduction. No strength
    sweep is performed here. Zero strength and K1 directly use historical DDIM.
    """
    if strength < 0 or bandwidth <= 0:
        raise ValueError("nonnegative strength and positive bandwidth required")
    if strength > 0 and getattr(schedule, "parameterization", "epsilon") != "epsilon":
        raise ValueError("nonzero PG currently supports epsilon prediction only")
    if strength == 0 or k == 1:
        return schedule.sample(model, condition, k, steps, generator)
    was_training = model.training
    wrapper = _GuidedDenoiser(model, schedule, strength, bandwidth)
    try:
        return schedule.sample(wrapper, condition, k, steps, generator)
    finally:
        model.train(was_training)
