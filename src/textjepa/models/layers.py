"""Shared building blocks: MLPs, token transformer, masked pooling."""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(maxsize=1)
def _external_flash_functions():
    """Optional FlashAttention varlen kernels, without a hard dependency."""
    fa4 = fa2 = None
    try:
        from flash_attn.cute import flash_attn_varlen_func
        fa4 = flash_attn_varlen_func
    except (ImportError, OSError):
        pass
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
        fa2 = flash_attn_varlen_qkvpacked_func
    except (ImportError, OSError):
        pass
    return fa4, fa2


class PackedMultiheadAttention(nn.MultiheadAttention):
    """Checkpoint-compatible attention with padding-free varlen execution."""

    VALID_BACKENDS = {
        "auto", "flash_attn_4", "flash_attn_2", "torch_sdpa", "torch_flex",
    }

    def __init__(self, *args, attention_backend="auto", **kwargs):
        super().__init__(*args, **kwargs)
        if attention_backend not in self.VALID_BACKENDS:
            raise ValueError(f"unknown packed attention backend: {attention_backend}")
        if not self.batch_first:
            raise ValueError("packed attention requires batch_first=True")
        self.attention_backend = attention_backend
        self.last_backend = "not_run"

    def _select_backend(self, query):
        if self.attention_backend != "auto":
            return self.attention_backend
        if query.is_cuda and query.dtype in {torch.float16, torch.bfloat16}:
            fa4, fa2 = _external_flash_functions()
            major, _ = torch.cuda.get_device_capability(query.device)
            if major >= 9 and fa4 is not None:
                return "flash_attn_4"
            if major >= 8 and fa2 is not None:
                return "flash_attn_2"
        # Native scaled-dot-product attention dispatches to PyTorch's fused
        # FlashAttention kernel.  Length bucketing below avoids a global
        # quadratic FlexAttention block mask when flash-attn is not installed.
        return "torch_sdpa"

    def _sdpa_packed(self, qkv, cu_seqlens, causal):
        """Execute compact right-padded length buckets with fused SDPA."""
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # A width of 16 bounds residual padding to 15 positions while keeping
        # the number of GPU kernel launches modest for all-start rollouts.
        buckets = ((lengths + 15) // 16) * 16
        output = qkv.new_empty((qkv.shape[0], self.num_heads, self.head_dim))
        for width_tensor in buckets.unique(sorted=True):
            width = int(width_tensor)
            rows = (buckets == width_tensor).nonzero(as_tuple=False).flatten()
            row_lengths = lengths[rows]
            offset = torch.arange(width, device=qkv.device)[None]
            valid = offset < row_lengths[:, None]
            indices = cu_seqlens[rows, None] + offset
            safe_indices = indices.clamp_max(qkv.shape[0] - 1)
            padded = qkv[safe_indices]
            q, k, v = (
                part.transpose(1, 2) for part in padded.unbind(2)
            )
            if causal:
                mask = None
            else:
                # Non-causal valid queries must not attend right padding.
                mask = valid[:, None, None, :]
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=causal,
            ).transpose(1, 2)
            output[indices[valid]] = attended[valid]
        return output

    def forward_packed(self, query, cu_seqlens, max_seqlen, block_mask, causal):
        if query.ndim != 2:
            raise ValueError("packed attention expects [tokens, dimension]")
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias).view(
            query.shape[0], 3, self.num_heads, self.head_dim
        )
        backend = self._select_backend(query)
        fa4, fa2 = _external_flash_functions()
        if backend == "flash_attn_4":
            if fa4 is None:
                raise RuntimeError("FlashAttention-4 is not installed")
            attended = fa4(
                qkv[:, 0], qkv[:, 1], qkv[:, 2],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=causal,
            )
        elif backend == "flash_attn_2":
            if fa2 is None:
                raise RuntimeError("FlashAttention-2 is not installed")
            attended = fa2(
                qkv, cu_seqlens, max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
            )
        elif backend == "torch_sdpa":
            attended = self._sdpa_packed(qkv, cu_seqlens, causal)
        elif query.is_cuda:
            if self.dropout:
                raise ValueError("FlexAttention packing requires dropout=0")
            from torch.nn.attention.flex_attention import flex_attention
            q, k, v = (
                part.transpose(0, 1).unsqueeze(0) for part in qkv.unbind(1)
            )
            attended = flex_attention(q, k, v, block_mask=block_mask)
            attended = attended.squeeze(0).transpose(0, 1)
        else:
            pieces = []
            for start, stop in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                lo, hi = int(start), int(stop)
                q, k, v = (
                    part[lo:hi].transpose(0, 1).unsqueeze(0)
                    for part in qkv.unbind(1)
                )
                value = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=causal,
                )
                pieces.append(value.squeeze(0).transpose(0, 1))
            attended = torch.cat(pieces)
        self.last_backend = f"{backend}_packed"
        return F.linear(
            attended.reshape(query.shape[0], self.embed_dim),
            self.out_proj.weight, self.out_proj.bias,
        )


def _packed_block_mask(lengths, total, device, causal):
    """FlexAttention block mask separating examples and causal prefixes."""
    from torch.nn.attention.flex_attention import create_block_mask
    padded = ((total + 127) // 128) * 128
    sequence = torch.full((padded,), -1, dtype=torch.int32, device=device)
    position = torch.full((padded,), -1, dtype=torch.int32, device=device)
    sequence[:total] = torch.repeat_interleave(
        torch.arange(len(lengths), device=device, dtype=torch.int32),
        lengths.long(),
    )
    position[:total] = torch.cat([
        torch.arange(int(length), device=device, dtype=torch.int32)
        for length in lengths
    ])

    def allowed(batch, head, query_index, key_index):
        same = (
            (query_index < total) & (key_index < total)
            & (sequence[query_index] == sequence[key_index])
        )
        return same & ((position[query_index] >= position[key_index]) if causal else True)

    return create_block_mask(
        allowed, B=1, H=None, Q_LEN=total, KV_LEN=total,
        device=device, _compile=False,
    )


def packed_encoder_forward(encoder, src, valid, causal=False):
    """Run attention and feed-forward blocks only on non-padding tokens."""
    if src.ndim != 3 or valid.shape != src.shape[:2]:
        raise ValueError("packed encoder expects [batch, length, dim] and mask")
    lengths = valid.sum(1, dtype=torch.int32)
    if (lengths == 0).any():
        raise ValueError("packed encoder does not accept empty sequences")
    flat = src[valid]
    cu = F.pad(lengths.cumsum(0), (1, 0))
    maximum = int(lengths.max())
    backend = encoder.layers[0].self_attn._select_backend(flat)
    block_mask = (
        _packed_block_mask(lengths, len(flat), flat.device, causal)
        if flat.is_cuda and backend == "torch_flex" else None
    )
    for layer in encoder.layers:
        if not layer.norm_first or not isinstance(
            layer.self_attn, PackedMultiheadAttention
        ):
            raise ValueError("packed execution requires pre-norm packed attention")
        attended = layer.self_attn.forward_packed(
            layer.norm1(flat), cu, maximum, block_mask, causal
        )
        flat = flat + layer.dropout1(attended)
        flat = flat + layer._ff_block(layer.norm2(flat))
    if encoder.norm is not None:
        flat = encoder.norm(flat)
    output = src.new_zeros(src.shape)
    output[valid] = flat
    return output


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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pad = tokens.eq(self.pad_id)
        key_pad = pad.clone()
        key_pad[pad.all(dim=-1), 0] = False  # keep all-pad rows finite
        h = self.encoder(
            self.tok(tokens) + self.pos[:, : tokens.shape[1]],
            src_key_padding_mask=key_pad,
        )
        keep = (~pad).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.norm(pooled)


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


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    """Compact causal mask shared across a batch (True = blocked).

    For right-padded token streams no key-padding mask is required: a valid
    query can only see earlier valid tokens, while padded query outputs are
    discarded.  Keeping this mask two-dimensional avoids the former
    B*H-fold expansion and leaves attention eligible for fused SDPA kernels.
    """
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
