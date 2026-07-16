"""Objective protocol and weighted composition.

Each objective maps (model outputs, batch) -> scalar loss. New losses are
added by writing a new module and listing it in the objective config —
nothing existing changes (open-closed).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Objective(nn.Module):
    def forward(self, out, batch: dict) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


def latent_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "smooth_l1",
    norm_targets: bool = True,
) -> torch.Tensor:
    """Elementwise distance [..] -> [..] (mean over feature dim)."""
    if norm_targets:
        pred = F.layer_norm(pred, pred.shape[-1:])
        target = F.layer_norm(target, target.shape[-1:])
    if kind == "l1":
        return (pred - target).abs().mean(-1)
    if kind == "mse":
        return (pred - target).pow(2).mean(-1)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1)
    raise ValueError(f"unknown distance {kind}")


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


class CompositeObjective(nn.Module):
    """Weighted sum of named objectives; returns (total, per-loss dict)."""

    def __init__(self, objectives: dict[str, Objective], weights: dict[str, float]):
        super().__init__()
        self.objectives = nn.ModuleDict(objectives)
        self.weights = dict(weights)

    def forward(self, out, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        total, items = 0.0, {}
        for name, obj in self.objectives.items():
            if self.weights.get(name, 1.0) == 0.0:
                items[name] = 0.0
                continue
            loss = obj(out, batch)
            total = total + self.weights.get(name, 1.0) * loss
            items[name] = loss.detach().item()
        return total, items
