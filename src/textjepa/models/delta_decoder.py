"""Delta-JEPA latent-difference action decoder (LDAD).

Decodes the executed action from the latent displacement
``delta = s_{t+1} - s_t`` alone. Displacement-level supervision keeps
transitions action-sensitive and prevents adjacent-state collapse
(arXiv:2606.31232) without any text reconstruction: targets are the op
class and the EMA embedding of the action phrase.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import mlp


class DeltaActionDecoder(nn.Module):
    def __init__(self, d_state: int, d_model: int, n_ops: int = 4, hidden_mult: int = 2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_state),
            nn.Linear(d_state, d_state * hidden_mult),
            nn.GELU(),
        )
        self.op_head = nn.Linear(d_state * hidden_mult, n_ops)
        self.emb_head = mlp([d_state * hidden_mult], d_model)

    def forward(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(delta)
        return self.op_head(h), self.emb_head(h)
