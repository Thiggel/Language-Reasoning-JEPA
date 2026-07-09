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

    @torch.no_grad()
    def update(self, online: nn.Module, momentum: float) -> None:
        for pt, po in zip(self.module.parameters(), online.parameters()):
            pt.lerp_(po, 1.0 - momentum)
        for bt, bo in zip(self.module.buffers(), online.buffers()):
            bt.copy_(bo)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
