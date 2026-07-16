"""Decoder-only transformer LM baseline (token-level, teacher-forced).

The comparison class for the JEPA world model: same data, same vocab,
trained to model the trace text autoregressively.  It supports both an
information-matched intent-policy evaluation and the historical diagnostic
that scores rendered candidate outcomes (which contain privileged values).
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import build_causal_attention_mask


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 288,
        n_layers: int = 8,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.n_heads = n_heads
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, L] -> [B, L, V] next-token logits."""
        B, L = tokens.shape
        x = self.tok(tokens) + self.pos[:, :L]
        valid = tokens != self.pad_id
        mask = build_causal_attention_mask(valid, self.n_heads)
        x = self.blocks(x, mask=mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def sequence_logprob(
        self, tokens: torch.Tensor, score_from: torch.Tensor
    ) -> torch.Tensor:
        """Sum log p(token_i | tokens_<i) for positions i >= score_from[b],
        ignoring pads. tokens [B, L], score_from [B] -> [B]."""
        logits = self.forward(tokens)  # [B, L, V]
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = tokens[:, 1:]
        pick = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        pos = torch.arange(tgt.shape[1], device=tokens.device).unsqueeze(0)
        mask = (pos >= (score_from.unsqueeze(1) - 1)) & (tgt != self.pad_id)
        return (pick * mask).sum(dim=1)
