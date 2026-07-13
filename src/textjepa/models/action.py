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


class VariationalAction(nn.Module):
    """Unobserved latent actions: posterior q(a | s_prev, s_next) infers
    the action code from the observed transition; prior p(a | s_prev)
    proposes codes at plan time. Reparametrized Gaussian, 16-d."""

    def __init__(self, d_state: int, d_action: int, hidden: int = 256):
        super().__init__()
        self.post = mlp([2 * d_state, hidden], 2 * d_action)
        self.prior = mlp([d_state, hidden], 2 * d_action)

    @staticmethod
    def _split(x):
        mu, logvar = x.chunk(2, dim=-1)
        return mu, logvar.clamp(-6, 2)

    def sample_posterior(self, s_prev, s_next):
        mu, logvar = self._split(self.post(torch.cat([s_prev, s_next], -1)))
        a = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return a, (mu, logvar)

    def prior_params(self, s_prev):
        return self._split(self.prior(s_prev))

    def sample_prior(self, s_prev, k: int = 1):
        mu, logvar = self.prior_params(s_prev)
        std = (0.5 * logvar).exp()
        if k == 1:
            return mu + torch.randn_like(mu) * std
        return mu.unsqueeze(-2) + torch.randn(
            *mu.shape[:-1], k, mu.shape[-1], device=mu.device
        ) * std.unsqueeze(-2)

    @staticmethod
    def kl(q_params, p_params):
        qm, ql = q_params
        pm, pl = p_params
        return 0.5 * (
            pl - ql + (ql.exp() + (qm - pm) ** 2) / pl.exp() - 1
        ).sum(-1)
