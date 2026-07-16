"""Sentence-latent LM baseline: one latent per sentence, causal
next-sentence prediction WITH a token decoder (reconstruction-based).

Two prediction targets:
- outcome (default): predict each rendered next-step sentence;
- intent: predict only interleaved action chunks and observe outcome chunks.

Two model modes:
- decoder-only (``latent_target=False``): CE of next-sentence tokens
  decoded from the context latent (a "chunked LM").
- semi-JEPA (``latent_target=True``): + regression of the context latent
  onto the encoded next-sentence latent (stopgrad) — latent prediction
  and reconstruction combined.

Isolates which JEPA ingredient matters: sentence-level abstraction
(both variants have it), latent prediction targets (only the second),
reconstruction-freeness (neither).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from textjepa.models.layers import TokenTransformer, mlp
from textjepa.models.state_model import DiscourseStateModel


class SentenceLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        chunk_layers: int = 2,
        chunk_heads: int = 4,
        state_layers: int = 4,
        state_heads: int = 8,
        dec_layers: int = 2,
        dec_heads: int = 4,
        ff_mult: int = 4,
        max_chunk_len: int = 48,
        max_chunks: int = 64,
        latent_target: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.latent_target = latent_target
        self.chunk_encoder = TokenTransformer(
            vocab_size, pad_id, d_model, chunk_layers, chunk_heads,
            ff_mult, max_chunk_len, 0.0,
        )
        self.state_model = DiscourseStateModel(
            d_model, state_layers, state_heads, ff_mult, max_chunks, 0.0
        )
        self.dec_tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.dec_tok.weight, std=0.02)
        self.dec_pos = nn.Parameter(torch.zeros(1, max_chunk_len, d_model))
        nn.init.normal_(self.dec_pos, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model, dec_heads, d_model * ff_mult, 0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            layer, dec_layers, norm=nn.LayerNorm(d_model)
        )
        self.dec_head = nn.Linear(d_model, vocab_size, bias=False)
        self.dec_head.weight = self.dec_tok.weight
        self.latent_head = mlp([d_model, d_model * 2], d_model)

    def encode_chunks(self, tokens: torch.Tensor) -> torch.Tensor:
        B, C, L = tokens.shape
        return self.chunk_encoder(tokens.reshape(B * C, L)).reshape(B, C, -1)

    def contexts(self, batch: dict) -> torch.Tensor:
        """[B, T, D]: context latent that must predict step t (s0 for t=0)."""
        prompt_emb = self.encode_chunks(batch["prompt_tokens"])
        step_emb = self.encode_chunks(batch["step_tokens"])
        s0, states = self.state_model(
            prompt_emb, batch["prompt_mask"], step_emb, batch["step_mask"]
        )
        return torch.cat([s0.unsqueeze(1), states[:, :-1]], dim=1)

    def decode_ce(
        self, ctx: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Per-sequence CE of teacher-forced token decoding from a single
        context latent. ctx [N, D], tokens [N, L] -> [N]."""
        N, L = tokens.shape
        inp = torch.cat(
            [torch.full((N, 1), self.pad_id, device=tokens.device,
                        dtype=torch.long), tokens[:, :-1]], dim=1
        )
        x = self.dec_tok(inp) + self.dec_pos[:, :L]
        causal = nn.Transformer.generate_square_subsequent_mask(
            L, device=tokens.device
        )
        h = self.decoder(x, ctx.unsqueeze(1), tgt_mask=causal)
        logits = self.dec_head(h)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1),
            ignore_index=self.pad_id, reduction="none",
        ).reshape(N, L)
        return ce.sum(dim=1)

    def forward(self, batch: dict) -> dict:
        ctx = self.contexts(batch)  # [B, T, D]
        B, T, D = ctx.shape
        m = batch.get("target_mask", batch["step_mask"]).reshape(-1)
        ctx_flat = ctx.reshape(-1, D)[m]
        tok_flat = batch["step_tokens"].reshape(B * T, -1)[m]
        ce = self.decode_ce(ctx_flat, tok_flat)
        n_tok = (tok_flat != self.pad_id).sum(dim=1).clamp(min=1)
        out = {"ce": (ce / n_tok).mean()}
        if self.latent_target:
            with torch.no_grad():
                tgt = self.encode_chunks(batch["step_tokens"])
            pred = self.latent_head(ctx)
            ln = lambda x: F.layer_norm(x, x.shape[-1:])
            err = F.smooth_l1_loss(ln(pred), ln(tgt), reduction="none").mean(-1)
            out["latent"] = (err.reshape(-1)[m]).mean()
        return out
