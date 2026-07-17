"""EMA teacher wrapper for JEPA target encoders."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMATeacher(nn.Module):
    def __init__(self, online: nn.Module):
        super().__init__()
        self.module = copy.deepcopy(online)
        self.module.requires_grad_(False)
        self.module.eval()

    def train(self, mode: bool = True):
        """Keep target networks in evaluation mode under ``parent.train()``.

        EMA targets are deterministic inference networks.  Calling
        ``model.train()`` recursively must never reactivate dropout, batch
        statistics, or any future training-only encoder behaviour.
        """
        super().train(False)
        self.module.eval()
        return self

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        for pt, po in zip(self.module.parameters(), online.parameters()):
            pt.lerp_(po, 1.0 - momentum)
        for bt, bo in zip(self.module.buffers(), online.buffers()):
            bt.copy_(bo)
        self.module.eval()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
