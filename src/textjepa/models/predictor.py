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


class AttnEditPredictor(nn.Module):
    """Edit-track attention predictor: F(sentence_embs, mask, s, a).

    The pooled slot state cannot represent WHICH sentence an edit changes
    (audit matching stuck at 0.44); here the predictor cross-attends over
    the current buffer's per-sentence embeddings with action-conditioned
    queries, then outputs the next pooled state."""

    def __init__(self, d_state: int, d_action: int, n_heads: int = 4,
                 n_layers: int = 2, n_queries: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_state))
        nn.init.normal_(self.queries, std=0.02)
        self.a_proj = nn.Linear(d_action, d_state)
        self.s_proj = nn.Linear(d_state, d_state)
        layer = nn.TransformerDecoderLayer(
            d_state, n_heads, d_state * 4, 0.0, "gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, n_layers, norm=nn.LayerNorm(d_state)
        )
        self.out = nn.Linear(n_queries * d_state, d_state)

    def forward(
        self,
        sent_emb: torch.Tensor,   # [N, C, D] current buffer sentences
        sent_mask: torch.Tensor,  # [N, C]
        s: torch.Tensor,          # [N, D] pooled current state
        a: torch.Tensor,          # [N, d_action]
    ) -> torch.Tensor:
        q = self.queries + self.a_proj(a).unsqueeze(1) + self.s_proj(s).unsqueeze(1)
        key_pad = ~sent_mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(dim=-1), 0] = False
        h = self.decoder(q.expand(-1, self.queries.shape[1], -1)
                         if q.shape[1] == 1 else q, sent_emb,
                         memory_key_padding_mask=key_pad)
        return s + self.out(h.flatten(1))
