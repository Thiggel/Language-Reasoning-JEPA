"""Action-conditioned latent predictors."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import mlp


class ActionConditionedPredictor(nn.Module):
    """s_hat_{t+1} = s_t + MLP([LN(s_t); a_t]) — residual latent dynamics."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden_mult: int = 4,
        n_hidden_layers: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(d_state)
        dims = [d_state + d_action] + [d_state * hidden_mult] * n_hidden_layers
        self.net = mlp(dims, d_state)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([self.norm(s), a], dim=-1))
        return s + out if self.residual else out


class FiLMPredictor(nn.Module):
    """Trunk-conditioned variant: the action modulates every hidden layer
    via FiLM (scale/shift), instead of one-shot input concatenation —
    the MLP analog of AdaLN trunk conditioning."""

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden_mult: int = 4,
        n_hidden_layers: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.norm = nn.LayerNorm(d_state)
        d_h = d_state * hidden_mult
        self.layers = nn.ModuleList()
        self.films = nn.ModuleList()
        d_in = d_state
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(d_in, d_h))
            self.films.append(nn.Linear(d_action, 2 * d_h))
            d_in = d_h
        self.out = nn.Linear(d_in, d_state)
        self.act = nn.GELU()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.norm(s)
        for layer, film in zip(self.layers, self.films):
            gamma, beta = film(a).chunk(2, dim=-1)
            h = self.act((1 + gamma) * layer(h) + beta)
        out = self.out(h)
        return s + out if self.residual else out
