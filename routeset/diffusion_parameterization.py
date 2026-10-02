"""Standard epsilon/v parameterizations without modifying historical diffusion.

v follows Salimans & Ho (ICLR2022), with direct stable x0/epsilon conversion as
in the official Diffusers DDIM scheduler. This is a baseline training repair,
not a new route-set mechanism. Epsilon sampling delegates to the original code.
"""
import torch

from .diffusion import DiffusionSchedule


class ParameterizedDiffusionSchedule(DiffusionSchedule):
    def __init__(self, steps=100, device="cpu", clip_x0=2., parameterization="epsilon"):
        super().__init__(steps, device, clip_x0)
        if parameterization not in ("epsilon", "v"):
            raise ValueError("parameterization must be epsilon or v")
        self.parameterization = parameterization

    def training_target(self, clean, noise, times):
        if self.parameterization == "epsilon":
            return noise
        alpha = self._extract(self.alpha_bars, times, clean)
        return alpha.sqrt() * noise - (1 - alpha).sqrt() * clean

    def prediction_to_x0_epsilon(self, noisy, times, prediction):
        alpha = self._extract(self.alpha_bars, times, noisy)
        if self.parameterization == "epsilon":
            return self.x0_from_eps(noisy, times, prediction), prediction
        # Do not convert v to epsilon then divide by sqrt(alpha): that would
        # reintroduce the high-noise cancellation that this control tests.
        return (alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction,
                alpha.sqrt() * prediction + (1 - alpha).sqrt() * noisy)

    @torch.no_grad()
    def sample(self, model, cond, k, steps=50, generator=None, eta=0.):
        if self.parameterization == "epsilon":
            return super().sample(model, cond, k, steps, generator, eta)
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
                times = timestep.expand(cond.shape[0])
                prediction = model(sample, self.normalized_time(times), cond)
                predicted_x0, epsilon = self.prediction_to_x0_epsilon(sample, times, prediction)
                alpha = alpha_bars[timestep]
                previous_alpha = alpha_bars[indices[position + 1]] if position + 1 < steps else sample.new_tensor(1.)
                if self.clip_x0 is not None:
                    predicted_x0 = predicted_x0.clamp(-self.clip_x0, self.clip_x0)
                sigma = eta * ((1 - previous_alpha) / (1 - alpha) * (1 - alpha / previous_alpha)).clamp_min(0).sqrt()
                direction = (1 - previous_alpha - sigma.square()).clamp_min(0).sqrt() * epsilon
                sample = previous_alpha.sqrt() * predicted_x0 + direction
                if eta and position + 1 < steps:
                    sample = sample + sigma * torch.randn(shape, dtype=cond.dtype, device=cond.device, generator=generator)
        finally:
            model.train(was_training)
        return sample
