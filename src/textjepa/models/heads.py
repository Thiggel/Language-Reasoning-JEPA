"""Goal/value heads scoring latent states against the prompt."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import mlp


class ValueHead(nn.Module):
    """Predicts remaining necessary steps from (state, goal-state) pairs.

    Serves as the goal energy at planning time: lower predicted remaining
    steps = closer to the solution region for this prompt.
    """

    def __init__(self, d_state: int, hidden_mult: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_state),
            mlp([2 * d_state, d_state * hidden_mult], 1),
        )

    def forward(self, s: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        s0 = s0.unsqueeze(-2).expand_as(s) if s.dim() > s0.dim() else s0
        return self.net(torch.cat([s, s0], dim=-1)).squeeze(-1)
