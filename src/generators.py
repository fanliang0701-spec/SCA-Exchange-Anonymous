"""Recorded FM4/DDPM16 model and sampler kernels."""
import math, time
import torch
from torch import nn
from torch.nn import functional as F

def label_features(y, dtype=None):
    """Eight LSB-first +/-1 bits and (Hamming weight - 4) / sqrt(2)."""
    bits = ((y.long()[:, None] >> torch.arange(8, device=y.device)) & 1).to(
        dtype=dtype or torch.float32)
    return torch.cat((2 * bits - 1, (bits.sum(1, keepdim=True) - 4) / math.sqrt(2)), 1)


def cosine_schedule(steps=1000):
    t = torch.arange(steps + 1, dtype=torch.float64) / steps
    alpha = torch.cos((t + .008) / 1.008 * math.pi / 2).square()
    alpha = alpha / alpha[0]
    beta = (1 - alpha[1:] / alpha[:-1]).clamp(max=.999)
    return (1 - beta).cumprod(0).float()


class ConditionalGenerator(nn.Module):
    def __init__(self, kind, dim=32):
        super().__init__()
        if kind not in ("fm", "ddpm", "reflow"):
            raise ValueError("kind must be fm, ddpm, or reflow")
        self.kind, self.dim = kind, dim
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.net = nn.Sequential(nn.Linear(dim + 32 + 9, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, dim))
        self.register_buffer("alpha_bar", cosine_schedule() if kind == "ddpm" else torch.empty(0))
        self.register_buffer("clip_min", torch.full((dim,), -float("inf")))
        self.register_buffer("clip_max", torch.full((dim,), float("inf")))
        for name in ("validation_noise", "validation_times", "pair_noise", "pair_endpoints",
                     "validation_endpoints"):
            self.register_buffer(name, torch.empty(0))
        for name in ("pair_labels", "validation_labels"):
            self.register_buffer(name, torch.empty(0, dtype=torch.long))

    def forward(self, z, t, labels):
        phase = 2 * math.pi * t.reshape(-1, 1) * self.frequencies
        time_features = torch.cat((phase.sin(), phase.cos()), 1)
        return self.net(torch.cat((z, time_features.to(z.dtype), label_features(labels, z.dtype)), 1))


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def sample(model, kind, noise, labels, steps):
    """Euler flow integration or eta=0 DDIM, with exactly `steps` network calls."""
    if steps < 1 or kind not in ("fm", "ddpm", "reflow"):
        raise ValueError("a valid generator kind and positive steps are required")
    if noise.ndim != 2 or labels.shape != (len(noise),):
        raise ValueError("noise must be (N,D) and labels (N,)")
    if kind == "ddpm" and steps > len(model.alpha_bar):
        raise ValueError("DDIM steps cannot exceed the stored diffusion schedule")
    _synchronize(noise.device)
    started = time.perf_counter()
    was_training = model.training
    model.eval()
    z = noise.clone()
    clipped = torch.zeros((), device=z.device, dtype=torch.long)
    try:
        if kind in ("fm", "reflow"):
            grid = [i / steps for i in range(steps)]
            for t in grid:
                z = z + model(z, z.new_full((len(z),), t), labels) / steps
        else:
            grid = torch.linspace(len(model.alpha_bar) - 1, 0, steps).round().long().tolist()
            for i, timestep in enumerate(grid):
                alpha = model.alpha_bar[timestep].to(z.dtype)
                epsilon = model(z, z.new_full((len(z),), timestep / (len(model.alpha_bar) - 1)), labels)
                x0 = (z - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
                if hasattr(model, "clip_min"):
                    bounded = torch.maximum(torch.minimum(x0, model.clip_max), model.clip_min)
                    clipped += (bounded != x0).sum()
                    x0 = bounded
                previous = model.alpha_bar[grid[i + 1]].to(z.dtype) if i + 1 < steps else z.new_tensor(1.)
                z = previous.sqrt() * x0 + (1 - previous).sqrt() * epsilon
        _synchronize(noise.device)
        elapsed = time.perf_counter() - started
        model.sampling_info = {"nfe": steps, "examples": len(noise),
                               "network_example_evaluations": len(noise) * steps,
                               "seconds": elapsed, "timesteps": grid,
                               "clipped_coordinate_fraction": int(clipped) / max(1, noise.numel() * steps)}
        return z
    finally:
        model.train(was_training)


def _rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def _loss(model, real, labels, noise, times):
    if model.kind == "ddpm":
        alpha = model.alpha_bar[times.long(), None]
        state = alpha.sqrt() * real + (1 - alpha).sqrt() * noise
        target, network_time = noise, times / (len(model.alpha_bar) - 1)
    else:
        state = (1 - times[:, None]) * noise + times[:, None] * real
        target, network_time = real - noise, times
    return F.mse_loss(model(state, network_time, labels), target)

