"""Latent-difference action decoders.

``ObservedActionDecoder`` is the faithful text analogue of Delta-JEPA LDAD:
it reconstructs the externally observed action phrase from
``s_{t+1} - s_t`` alone.  ``DeltaActionDecoder`` is the repository's older
hybrid diagnostic, which predicts an operation class and a learned phrase
embedding and must not be presented as faithful raw-action LDAD.
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


class ObservedActionDecoder(nn.Module):
    """Decode the complete observed action-token sequence from a displacement.

    Learnable position queries are conditioned only on the latent difference;
    the decoder never sees either endpoint or the action encoder output.  Token
    cross-entropy is the discrete-text counterpart of Delta-JEPA's raw-action
    reconstruction loss.
    """

    def __init__(
        self,
        d_state: int,
        vocab_size: int,
        max_len: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.queries = nn.Parameter(torch.empty(1, self.max_len, d_state))
        nn.init.normal_(self.queries, std=0.02)
        self.condition = nn.Sequential(
            nn.LayerNorm(d_state), nn.Linear(d_state, d_state)
        )
        layer = nn.TransformerEncoderLayer(
            d_state, n_heads, d_state * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_state)
        )
        self.token_head = nn.Linear(d_state, vocab_size)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        shape = delta.shape[:-1]
        flat = delta.reshape(-1, delta.shape[-1])
        queries = self.queries + self.condition(flat).unsqueeze(1)
        logits = self.token_head(self.decoder(queries))
        return logits.reshape(*shape, self.max_len, logits.shape[-1])


class MultiStepObservedActionDecoder(nn.Module):
    """Decode an ordered sequence of observed action phrases from one delta.

    This is the discrete-text counterpart of Delta-JEPA's multi-step LDAD:
    ``s_{t+H} - s_t`` is the decoder's only input and must reconstruct the H
    externally observed actions in order.  Separate action/token queries keep
    phrase boundaries explicit; displacement-conditioned scale and shift
    modulate every Transformer block (the text analogue of AdaLN).
    """

    def __init__(
        self,
        d_state: int,
        vocab_size: int,
        max_len: int,
        horizon: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.max_len = int(max_len)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("LDAD horizon must be positive")
        self.action_queries = nn.Parameter(
            torch.empty(1, self.horizon, 1, d_state)
        )
        self.token_queries = nn.Parameter(torch.empty(1, 1, self.max_len, d_state))
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.token_queries, std=0.02)
        self.delta_norm = nn.LayerNorm(d_state)
        self.blocks = nn.ModuleList()
        self.modulations = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                nn.TransformerEncoderLayer(
                    d_state, n_heads, d_state * 2, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True,
                )
            )
            modulation = nn.Linear(d_state, 2 * d_state)
            nn.init.zeros_(modulation.weight)
            nn.init.zeros_(modulation.bias)
            self.modulations.append(modulation)
        self.final_norm = nn.LayerNorm(d_state)
        self.token_head = nn.Linear(d_state, vocab_size)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        shape = delta.shape[:-1]
        flat = delta.reshape(-1, delta.shape[-1])
        n = flat.shape[0]
        h = (self.action_queries + self.token_queries).reshape(
            1, self.horizon * self.max_len, -1
        ).expand(n, -1, -1)
        condition = self.delta_norm(flat)
        for block, modulation in zip(self.blocks, self.modulations):
            scale, shift = modulation(condition).chunk(2, dim=-1)
            h = block((1.0 + scale.unsqueeze(1)) * h + shift.unsqueeze(1))
        logits = self.token_head(self.final_norm(h)).reshape(
            n, self.horizon, self.max_len, -1
        )
        return logits.reshape(*shape, self.horizon, self.max_len, logits.shape[-1])
