"""Causal sequence encoder from level-0 token latents to level-1 sentence latents.

The default `E0_to_1` in `HierarchicalLanguageJEPA` is a two-layer MLP applied
pointwise:

    RMSNorm -> Linear(d_token, 2*d_sentence) -> SiLU -> Linear(-> d_sentence)

so the sentence latent at a boundary can only contain whatever `E0` compressed
into that single boundary token's state. It never aggregates the step. That is
conspicuous next to `P0`/`P1` (contextual predictors with attention) and `A1`
(which does encode the token span): the state encoder is the one pointwise map
in the stack.

Measured consequence (2026-08-13, `measure_worker_selection_quality.py`): given
a fair 32-candidate menu and a perfect waypoint, ranking by exactly re-encoded
sentence states scores 0.323 against a 0.209 blind floor and a 0.635 ceiling --
real but weak signal, consistent with a representation that carries coarse
structure but not the arithmetic that separates a correct step from a plausible
wrong one.

This module reads the whole causal history of token latents, so the sentence
state at position t is a function of z0_{<=t} rather than of z0_t alone. It is
causal, so it can still be read out at any boundary and stays valid for
autoregressive planning.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import causal_attention_mask


class CausalSentenceEncoder(nn.Module):
    """Map [batch, time, d_token] level-0 latents to [batch, time, d_sentence].

    Position ``t`` of the output attends only to positions ``<= t``.
    """

    def __init__(
        self,
        d_token: int,
        d_sentence: int,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        max_len: int = 2048,
    ):
        super().__init__()
        if min(d_token, d_sentence, n_layers, n_heads, ff_mult, max_len) < 1:
            raise ValueError("causal sentence encoder sizes must be positive")
        if d_token % n_heads:
            raise ValueError("d_token must be divisible by n_heads")
        self.d_token = int(d_token)
        self.d_sentence = int(d_sentence)
        self.input_norm = nn.RMSNorm(d_token)
        self.position = nn.Parameter(torch.zeros(1, max_len, d_token))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_token,
            n_heads,
            d_token * ff_mult,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.output_norm = nn.RMSNorm(d_token)
        self.project = nn.Linear(d_token, d_sentence)

    def _positions(self, length: int) -> torch.Tensor:
        if length <= self.position.shape[1]:
            return self.position[:, :length]
        # Longer-than-trained sequences interpolate rather than fail, so
        # length-OOD evaluation stays runnable.
        return torch.nn.functional.interpolate(
            self.position.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(
                "causal sentence encoder needs [batch, time, d_token]; a "
                "pointwise call would silently mix independent items along "
                "the time axis"
            )
        if states.shape[-1] != self.d_token:
            raise ValueError("token latent width does not match the encoder")
        length = states.shape[1]
        value = self.input_norm(states) + self._positions(length).to(
            states.dtype
        )
        mask = causal_attention_mask(length, states.device)
        encoded = self.blocks(value, mask=mask, is_causal=True)
        return self.project(self.output_norm(encoded))

    @torch.no_grad()
    def encode_last(self, states: torch.Tensor) -> torch.Tensor:
        """Sentence latent at the final position of each sequence."""

        return self(states)[:, -1]
