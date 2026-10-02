"""Cosine DDPM training schedule and deterministic DDIM route sampling."""

import math
from typing import Optional

import torch
from torch import nn


class DiffusionSchedule(nn.Module):
    """Epsilon-prediction diffusion over normalized interior route residuals.

    Training timesteps are integers in [0, steps-1]. Model-facing timesteps are
    normalized to [0,1]. ``q_sample`` returns noisy residuals and target epsilon.
    DDIM sampling uses eta=0 by default, with fresh independent Gaussian noise
    per route. Candidate communication is entirely the denoiser's responsibility.
    """

    def __init__(
        self, steps: int = 100, device: str = "cpu", clip_x0: Optional[float] = 2.0
    ):
        super().__init__()
        if steps < 2:
            raise ValueError("at least two diffusion steps are required")
        self.steps = steps
        self.num_steps = steps
        self.clip_x0 = clip_x0
        times = torch.linspace(0, 1, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((times + 0.008) / 1.008) * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999).float()
        alphas = 1.0 - betas
        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", alphas.to(device))
        self.register_buffer("alpha_bars", alphas.cumprod(dim=0).to(device))

    def _extract(self, buffer: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        indices = timesteps.to(device=target.device, dtype=torch.long).reshape(-1)
        if indices.numel() != target.shape[0]:
            raise ValueError("one integer diffusion timestep is required per batch item")
        value = buffer.to(target.device)[indices].to(target.dtype)
        return value.reshape(target.shape[0], *([1] * (target.ndim - 1)))

    def normalized_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timesteps.float() / (self.num_steps - 1)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        if noise is None:
            noise = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0")
        alpha_bar = self._extract(self.alpha_bars, timesteps, x0)
        noisy = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    # Convenient synonym used in some training code.
    add_noise = q_sample

    def x0_from_eps(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, timesteps, noisy)
        return (noisy - (1.0 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt().clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond: torch.Tensor,
        k: int,
        steps: int = 50,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Return [B,K,model.horizon-2,3] normalized residuals.

        A generator makes initial noise reproducible. Sampling temporarily uses
        evaluation mode and restores the model's previous training state.
        ``steps`` is the number of denoiser evaluations (<= training steps).
        """
        if k < 1 or steps < 1 or eta < 0:
            raise ValueError("k and sampling steps must be positive; eta nonnegative")
        steps = min(int(steps), self.num_steps)
        shape = (cond.shape[0], k, model.horizon - 2, 3)
        sample = torch.randn(shape, dtype=cond.dtype, device=cond.device, generator=generator)
        indices = torch.linspace(self.num_steps - 1, 0, steps, device=cond.device).round().long()
        alpha_bars = self.alpha_bars.to(device=cond.device, dtype=cond.dtype)
        was_training = model.training
        model.eval()
        try:
            for position, timestep in enumerate(indices):
                t = timestep.expand(cond.shape[0])
                epsilon = model(sample, self.normalized_time(t), cond)
                alpha = alpha_bars[timestep]
                previous_alpha = alpha_bars[indices[position + 1]] if position + 1 < steps else sample.new_tensor(1.0)
                predicted_x0 = (sample - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1e-8)
                if self.clip_x0 is not None:
                    predicted_x0 = predicted_x0.clamp(-self.clip_x0, self.clip_x0)
                sigma = eta * ((1.0 - previous_alpha) / (1.0 - alpha) * (1.0 - alpha / previous_alpha)).clamp_min(0).sqrt()
                direction = (1.0 - previous_alpha - sigma.square()).clamp_min(0).sqrt() * epsilon
                sample = previous_alpha.sqrt() * predicted_x0 + direction
                if eta and position + 1 < steps:
                    sample = sample + sigma * torch.randn(shape, dtype=cond.dtype, device=cond.device, generator=generator)
        finally:
            model.train(was_training)
        return sample
