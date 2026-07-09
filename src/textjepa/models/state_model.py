"""Causal discourse-state model over chunk embeddings."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import build_causal_attention_mask, encoder_stack


class DiscourseStateModel(nn.Module):
    """Runs a causal transformer over [prompt chunks | step chunks].

    Returns ``s0`` (state at the question, i.e. the last valid prompt chunk)
    and ``step_states[t]`` = state after reasoning step ``t``.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_chunks: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.pos = nn.Parameter(torch.zeros(1, max_chunks, d_model))
        self.segment = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.segment, std=0.02)
        self.encoder = encoder_stack(d_model, n_layers, n_heads, ff_mult, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        prompt_emb: torch.Tensor,  # [B, P, D]
        prompt_mask: torch.Tensor,  # [B, P] bool
        step_emb: torch.Tensor,  # [B, T, D]
        step_mask: torch.Tensor,  # [B, T] bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, _ = prompt_emb.shape
        x = torch.cat(
            [prompt_emb + self.segment[0], step_emb + self.segment[1]], dim=1
        )
        x = x + self.pos[:, : x.shape[1]]
        valid = torch.cat([prompt_mask, step_mask], dim=1)
        attn_mask = build_causal_attention_mask(valid, self.n_heads)
        h = self.norm(self.encoder(x, mask=attn_mask))
        q_pos = prompt_mask.sum(dim=1) - 1
        s0 = h[torch.arange(B, device=h.device), q_pos]
        return s0, h[:, P:]
