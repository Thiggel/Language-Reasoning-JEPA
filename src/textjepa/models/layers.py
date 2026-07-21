"""Shared building blocks: MLPs, token transformer, masked pooling."""

from __future__ import annotations

import torch
from torch import nn


def mlp(dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def encoder_stack(
    d_model: int, n_layers: int, n_heads: int, ff_mult: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model,
        n_heads,
        dim_feedforward=d_model * ff_mult,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class LoopedTransformerEncoder(nn.Module):
    """A weight-shared Transformer block with a controlled loop schedule.

    During training, one loop count is sampled for the whole batch from a
    clipped shifted-Poisson distribution.  Evaluation is deterministic and
    uses ``eval_loops`` unless the caller explicitly supplies ``num_loops``.
    Sharing one block makes additional evaluation loops genuine test-time
    compute rather than additional trained parameters.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int,
        dropout: float = 0.0,
        train_loop_mean: float = 4.0,
        train_loop_min: int = 1,
        train_loop_max: int = 8,
        eval_loops: int = 4,
    ):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("looped reasoning baselines require dropout=0")
        if train_loop_min < 1 or train_loop_max < train_loop_min:
            raise ValueError("invalid loop-count bounds")
        if train_loop_mean < 1:
            raise ValueError("train_loop_mean must be at least one")
        if not train_loop_min <= eval_loops <= train_loop_max:
            raise ValueError("eval_loops must lie inside the training bounds")
        self.train_loop_mean = float(train_loop_mean)
        self.train_loop_min = int(train_loop_min)
        self.train_loop_max = int(train_loop_max)
        self.eval_loops = int(eval_loops)
        self.last_num_loops = self.eval_loops
        self.block = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def sample_num_loops(self) -> int:
        rate = max(self.train_loop_mean - 1.0, 0.0)
        sampled = 1 + int(torch.poisson(torch.tensor(rate)).item())
        return min(max(sampled, self.train_loop_min), self.train_loop_max)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> torch.Tensor:
        loops = (
            self.sample_num_loops() if self.training else self.eval_loops
        ) if num_loops is None else int(num_loops)
        if loops < 1:
            raise ValueError("num_loops must be positive")
        self.last_num_loops = loops
        for _ in range(loops):
            x = self.block(
                x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return x


class TokenTransformer(nn.Module):
    """Tokens [N, L] -> pooled chunk embedding [N, D] (masked mean)."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        max_len: int = 48,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.encoder = encoder_stack(d_model, n_layers, n_heads, ff_mult, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextual token latents and their validity mask."""
        pad = tokens.eq(self.pad_id)
        key_pad = pad.clone()
        key_pad[pad.all(dim=-1), 0] = False  # keep all-pad rows finite
        h = self.encoder(
            self.tok(tokens) + self.pos[:, : tokens.shape[1]],
            src_key_padding_mask=key_pad,
        )
        return self.norm(h), ~pad

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h, valid = self.forward_tokens(tokens)
        keep = valid.unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return pooled


def build_causal_attention_mask(
    key_valid: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """Bool attention mask [B*H, S, S] (True = blocked): causal + key padding.

    Rows left with no allowed key fall back to attending position 0 so that
    padded positions stay finite (their outputs are never read back).
    """
    B, S = key_valid.shape
    causal = torch.ones(S, S, dtype=torch.bool, device=key_valid.device).tril()
    allowed = causal.unsqueeze(0) & key_valid.unsqueeze(1)
    dead = ~allowed.any(dim=-1)
    allowed[..., 0] |= dead
    return (~allowed).repeat_interleave(n_heads, dim=0)
