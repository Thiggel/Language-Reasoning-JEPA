"""Action bottlenecks: compressed step codes and macro-actions.

The action latent is deliberately tiny (HWM finds ~4-8 dims optimal): it
must carry the *intent* of a discourse move, not its content.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import encoder_stack, mlp


class FSQ(nn.Module):
    """Finite scalar quantization with a straight-through estimator."""

    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z)
        half = (self.levels - 1) / 2
        zq = torch.round(z * half) / half
        return z + (zq - z).detach()


class ActionEncoder(nn.Module):
    """Phrase embedding [.., D] -> small action latent [.., d_action]."""

    def __init__(
        self,
        d_model: int,
        d_action: int = 16,
        hidden_mult: int = 2,
        fsq_levels: list[int] | None = None,
    ):
        super().__init__()
        self.proj = mlp([d_model, d_model * hidden_mult // 2], d_action)
        self.quantizer = FSQ(fsq_levels) if fsq_levels else None

    def forward(self, phrase_emb: torch.Tensor) -> torch.Tensor:
        a = self.proj(phrase_emb)
        return self.quantizer(a) if self.quantizer else a


class MacroActionEncoder(nn.Module):
    """CLS-transformer over K action latents -> macro-action [.., d_macro]."""

    def __init__(
        self,
        d_action: int,
        d_macro: int = 8,
        d_hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.inp = nn.Linear(d_action, d_hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_hidden))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, 16, d_hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.encoder = encoder_stack(d_hidden, n_layers, n_heads, 2, 0.0)
        self.out = nn.Linear(d_hidden, d_macro)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [N, K, d_action] -> [N, d_macro]."""
        h = self.inp(actions) + self.pos[:, 1 : actions.shape[1] + 1]
        h = torch.cat([self.cls.expand(h.shape[0], 1, -1) + self.pos[:, :1], h], 1)
        return self.out(self.encoder(h)[:, 0])
