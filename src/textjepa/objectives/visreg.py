"""Variance-Invariance-Sketching regularization (Wu et al., 2026)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class VISReg(nn.Module):
    """Faithful scale + center + sliced-Wasserstein shape regularizer.

    The official implementation consumes ``[views, batch, dimension]``.  A
    two-dimensional tensor is treated as one view, which is the natural form
    for TextJEPA's valid online states.
    """

    def __init__(self, num_projections: int = 4096):
        super().__init__()
        self.num_projections = int(num_projections)
        if self.num_projections < 1:
            raise ValueError("num_projections must be positive")
        self._target_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _target(self, batch: int, device, dtype):
        key = (int(batch), device, dtype)
        target = self._target_cache.get(key)
        if target is None:
            quantile = torch.linspace(
                1, batch, batch, device=device, dtype=torch.float32
            ) / (batch + 1)
            target = torch.erfinv(2 * quantile - 1).mul_(math.sqrt(2))
            target = target.to(dtype=dtype).view(1, batch, 1)
            self._target_cache[key] = target
        return target

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.ndim != 3:
            raise ValueError("VISReg expects [batch, dim] or [views, batch, dim]")
        _, batch, dimension = states.shape
        if batch < 2:
            raise ValueError("VISReg requires at least two samples")
        mean = states.mean(dim=1, keepdim=True)
        center = mean.square().mean()
        centered = states - mean
        std = centered.norm(dim=1).div(math.sqrt(batch)).add(1e-6)
        scale = (std - 1.0).square().mean()
        normalized = centered / std.detach().unsqueeze(1)
        projections = F.normalize(torch.randn(
            dimension, self.num_projections,
            device=states.device, dtype=states.dtype,
        ), dim=0)
        projected = (normalized @ projections).sort(dim=1).values
        shape = (projected - self._target(batch, states.device, states.dtype)).square().mean()
        return scale + shape + center
