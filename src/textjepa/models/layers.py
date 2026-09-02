"""Shared building blocks: MLPs, token transformer, masked pooling."""

from __future__ import annotations

from contextlib import nullcontext
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


@lru_cache(maxsize=1)
def _external_flash_functions():
    """Return optional FA4/FA2 varlen kernels without hard dependencies."""
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


def _bucketed_packed_attention(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dropout_p: float = 0.0,
    backend: str = "torch",
    causal: bool = False,
) -> torch.Tensor:
    """Native SDPA over isolated power-of-two length buckets.

    Input/output are flat ``[total_tokens, heads, head_dim]`` streams. Each
    sequence occupies its own SDPA batch row, so cross-example attention is
    structurally impossible. Bucket padding is always less than 2x.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    groups: dict[int, list[tuple[int, int]]] = {}
    for start, length in zip(cu_seqlens[:-1].tolist(), lengths):
        bucket = max(1, 1 << (int(length) - 1).bit_length())
        groups.setdefault(bucket, []).append((int(start), int(length)))
    heads, head_dim = qkv.shape[-2:]
    attended = qkv.new_zeros(qkv.shape[0], heads, head_dim)
    context = nullcontext()
    if qkv.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        allowed = (
            [SDPBackend.FLASH_ATTENTION]
            if backend == "torch_flash" else
            [SDPBackend.FLASH_ATTENTION,
             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        )
        context = sdpa_kernel(allowed)
    with context:
        for bucket, members in groups.items():
            starts = torch.tensor(
                [item[0] for item in members], device=qkv.device,
                dtype=torch.long,
            )
            member_lengths = torch.tensor(
                [item[1] for item in members], device=qkv.device,
                dtype=torch.long,
            )
            offset = torch.arange(bucket, device=qkv.device)
            gather_index = starts[:, None] + offset[None]
            group_valid = offset[None] < member_lengths[:, None]
            # Invalid bucket slots may point into the next sequence; mask them
            # immediately after the gather so neither values nor gradients leak.
            gathered = qkv[gather_index.clamp_max(qkv.shape[0] - 1)]
            gathered = gathered * group_valid[:, :, None, None, None]
            q_group, k_group, v_group = (
                part.permute(0, 2, 1, 3) for part in gathered.unbind(2)
            )
            # Sequences are right-padded within their bucket row, so under a
            # causal mask every key a valid query can see is itself valid and
            # no explicit padding mask is needed; padded rows are discarded by
            # the gather below.
            result = F.scaled_dot_product_attention(
                q_group, k_group, v_group,
                attn_mask=None if causal else group_valid[:, None, None, :],
                dropout_p=dropout_p, is_causal=causal,
            )
            unpacked = result.permute(0, 2, 1, 3)
            attended[gather_index[group_valid]] = unpacked[group_valid]
    return attended


class FlashMultiheadAttention(nn.MultiheadAttention):
    """Self-attention dispatched to FA4, FA2, or PyTorch fused SDPA.

    External FlashAttention uses its variable-length interface, so padding is
    removed before the quadratic kernel. The inherited projection parameter
    layout remains checkpoint-compatible with ``nn.MultiheadAttention``.
    """

    VALID_BACKENDS = {
        "auto", "flash_attn_4", "flash_attn_2", "torch_flash", "torch"
    }

    def __init__(self, *args, attention_backend: str = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        if attention_backend not in self.VALID_BACKENDS:
            raise ValueError(f"unknown attention backend: {attention_backend}")
        if not self.batch_first:
            raise ValueError("fused TextJEPA attention requires batch_first=True")
        self.attention_backend = attention_backend
        self.last_backend = "not_run"

    def _external_attention(self, qkv, valid, backend):
        fa4, fa2 = _external_flash_functions()
        flat = qkv[valid].contiguous()
        lengths = valid.sum(1, dtype=torch.int32)
        cu = F.pad(lengths.cumsum(0), (1, 0))
        maximum = int(lengths.max().item())
        if backend == "flash_attn_4":
            if fa4 is None:
                raise RuntimeError("FlashAttention-4 was requested but is not installed")
            output = fa4(
                flat[:, 0], flat[:, 1], flat[:, 2],
                cu_seqlens_q=cu, cu_seqlens_k=cu,
                max_seqlen_q=maximum, max_seqlen_k=maximum, causal=False,
            )
        else:
            if fa2 is None:
                raise RuntimeError("FlashAttention-2 was requested but is not installed")
            output = fa2(
                flat, cu, maximum,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )
        padded = output.new_zeros(
            qkv.shape[0], qkv.shape[1], self.num_heads, self.head_dim
        )
        padded[valid] = output
        self.last_backend = backend
        return padded

    def _select_backend(self, query):
        requested = self.attention_backend
        if requested != "auto":
            return requested
        if query.is_cuda and query.dtype in {torch.float16, torch.bfloat16}:
            fa4, fa2 = _external_flash_functions()
            major, _ = torch.cuda.get_device_capability(query.device)
            if major >= 9 and fa4 is not None:
                return "flash_attn_4"
            if major >= 8 and fa2 is not None:
                return "flash_attn_2"
            return "torch_flash"
        return "torch"

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                is_causal=False):
        if (query is not key or query is not value or need_weights
                or attn_mask is not None or is_causal or self.bias_k is not None
                or self.bias_v is not None or self.add_zero_attn):
            self.last_backend = "torch_mha_fallback"
            return super().forward(
                query, key, value, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                average_attn_weights=average_attn_weights, is_causal=is_causal,
            )
        batch, length, _ = query.shape
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        if key_padding_mask is None:
            valid = torch.ones(
                batch, length, dtype=torch.bool, device=query.device
            )
        elif key_padding_mask.dtype == torch.bool:
            valid = ~key_padding_mask
        else:
            # TransformerEncoder canonicalizes bool padding masks to additive
            # float masks (0 for valid, -inf for padding) before this call.
            valid = key_padding_mask.eq(0)
        backend = self._select_backend(query)
        if backend in {"flash_attn_4", "flash_attn_2"}:
            attended = self._external_attention(qkv, valid, backend)
        else:
            q, k, v = (part.transpose(1, 2) for part in qkv.unbind(2))
            mask = valid[:, None, None, :]
            context = nullcontext()
            if query.is_cuda:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                allowed = (
                    [SDPBackend.FLASH_ATTENTION]
                    if backend == "torch_flash" else
                    [SDPBackend.FLASH_ATTENTION,
                     SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
                )
                context = sdpa_kernel(allowed)
            with context:
                attended = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                ).transpose(1, 2)
            attended = attended * valid[:, :, None, None]
            self.last_backend = backend
        output = F.linear(
            attended.reshape(batch, length, self.embed_dim),
            self.out_proj.weight, self.out_proj.bias,
        )
        return output, None

    def forward_packed(
        self,
        query: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        block_mask=None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Self-attention over a flat, boundary-separated token stream.

        ``query`` contains no padding. ``cu_seqlens`` identifies independent
        examples; attention is therefore exactly block diagonal and can never
        leak information between packed training items. ``causal`` restricts
        each query to its own causal prefix within its example.
        """
        if query.ndim != 2:
            raise ValueError("packed attention expects [total_tokens, dim]")
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias).view(
            query.shape[0], 3, self.num_heads, self.head_dim
        )
        backend = self._select_backend(query)
        if backend == "flash_attn_4":
            fa4, _ = _external_flash_functions()
            if fa4 is None:
                raise RuntimeError("FlashAttention-4 was requested but is not installed")
            attended = fa4(
                qkv[:, 0], qkv[:, 1], qkv[:, 2],
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=causal,
            )
            self.last_backend = "flash_attn_4_packed"
        elif backend == "flash_attn_2":
            _, fa2 = _external_flash_functions()
            if fa2 is None:
                raise RuntimeError("FlashAttention-2 was requested but is not installed")
            attended = fa2(
                qkv, cu_seqlens, max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=causal,
            )
            self.last_backend = "flash_attn_2_packed"
        elif query.is_cuda:
            # Portable training fallback: keep the residual/MLP stream flat,
            # but batch attention by power-of-two sequence length. This uses
            # native fused SDPA, bounds attention slack below 2x, and never
            # constructs a cross-example score. External FA2/FA4 above remains
            # the completely padding-free path when installed.
            attended = _bucketed_packed_attention(
                qkv, cu_seqlens,
                dropout_p=self.dropout if self.training else 0.0,
                backend=backend, causal=causal,
            )
            self.last_backend = "torch_bucketed_packed"
        else:
            # Exact, deliberately simple CPU fallback for tests and debugging.
            pieces = []
            for start, stop in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                lo, hi = int(start.item()), int(stop.item())
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
            attended = torch.cat(pieces, dim=0)
            self.last_backend = "torch_loop_packed"
        return F.linear(
            attended.reshape(query.shape[0], self.embed_dim),
            self.out_proj.weight, self.out_proj.bias,
        )


def attention_backend_summary(module: nn.Module) -> list[str]:
    """Concrete kernels used by fused attention modules in the last forward."""
    return sorted({
        child.last_backend for child in module.modules()
        if isinstance(child, FlashMultiheadAttention)
    })


def packed_encoder_forward(
    encoder: nn.TransformerEncoder,
    src: torch.Tensor,
    valid: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Run a pre-norm encoder with no padded activations between layers.

    The returned tensor is dense only at the model boundary for compatibility
    with token-aligned losses and edit routing. All attention and feed-forward
    blocks operate on ``sum(valid)`` tokens. ``causal=True`` applies a causal
    mask within each packed example (used by the causal token/state stacks).
    """
    if src.ndim != 3 or valid.shape != src.shape[:2]:
        raise ValueError("packed encoder expects [batch, length, dim] plus mask")
    lengths = valid.sum(1, dtype=torch.int32)
    if (lengths == 0).any():
        raise ValueError("packed encoder does not accept empty sequences")
    flat = src[valid]
    actual_tokens = flat.shape[0]
    cu = F.pad(lengths.cumsum(0), (1, 0))
    maximum = int(lengths.max().item())
    for layer in encoder.layers:
        if not layer.norm_first:
            raise ValueError("packed encoder currently requires norm_first=True")
        if not isinstance(layer.self_attn, FlashMultiheadAttention):
            raise ValueError(
                "packed encoder requires a fused attention backend, not 'torch'"
            )
        attended = layer.self_attn.forward_packed(
            layer.norm1(flat), cu, maximum, causal=causal,
        )
        flat = flat + layer.dropout1(attended)
        flat = flat + layer._ff_block(layer.norm2(flat))
    if encoder.norm is not None:
        flat = encoder.norm(flat)
    output = src.new_zeros(src.shape)
    output[valid] = flat[:actual_tokens]
    return output


# Backward-compatible name used by the token-hierarchy modules.  The flash
# implementation subsumes the original packed-attention wrapper.
PackedMultiheadAttention = FlashMultiheadAttention


def mlp(dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


def encoder_stack(
    d_model: int, n_layers: int, n_heads: int, ff_mult: float, dropout: float,
    attention_backend: str = "torch",
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model,
        n_heads,
        dim_feedforward=int(round(d_model * ff_mult)),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    if attention_backend != "torch":
        layer.self_attn = FlashMultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
            attention_backend=attention_backend,
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
        train_loop_distribution: str = "shifted_poisson",
        train_loop_sigma: float = 0.5,
    ):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("looped reasoning baselines require dropout=0")
        if train_loop_min < 1 or train_loop_max < train_loop_min:
            raise ValueError("invalid loop-count bounds")
        if train_loop_mean < 1:
            raise ValueError("train_loop_mean must be at least one")
        if train_loop_distribution not in {
            "shifted_poisson", "poisson_lognormal"
        }:
            raise ValueError("unknown loop-count distribution")
        if train_loop_sigma < 0:
            raise ValueError("train_loop_sigma must be non-negative")
        if not train_loop_min <= eval_loops <= train_loop_max:
            raise ValueError("eval_loops must lie inside the training bounds")
        self.train_loop_mean = float(train_loop_mean)
        self.train_loop_min = int(train_loop_min)
        self.train_loop_max = int(train_loop_max)
        self.eval_loops = int(eval_loops)
        self.train_loop_distribution = train_loop_distribution
        self.train_loop_sigma = float(train_loop_sigma)
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
        target_rate = max(self.train_loop_mean - 1.0, 0.0)
        if self.train_loop_distribution == "poisson_lognormal" and target_rate:
            # Geiping et al. (2025), Eq. 1--2: choose the log-normal
            # location so E[rate] equals the requested pre-shift mean.
            sigma = self.train_loop_sigma
            log_rate = (
                torch.log(torch.tensor(target_rate))
                - 0.5 * sigma * sigma
                + sigma * torch.randn(())
            )
            rate = float(torch.exp(log_rate).item())
        else:
            rate = target_rate
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
        pos_kind: str = "learned",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.pos_kind = pos_kind
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        if pos_kind == "rope":
            self.pos = None
            self.encoder = RoPETransformerEncoder(
                d_model, n_heads, ff_mult, n_layers, dropout, bidirectional=True,
            )
        else:
            self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pos, std=0.02)
            self.encoder = encoder_stack(d_model, n_layers, n_heads, ff_mult, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextual token latents and their validity mask."""
        pad = tokens.eq(self.pad_id)
        key_pad = pad.clone()
        key_pad[pad.all(dim=-1), 0] = False  # keep all-pad rows finite
        x = self.tok(tokens)
        if self.pos is not None:
            x = x + self.pos[:, : tokens.shape[1]]
        h = self.encoder(x, src_key_padding_mask=key_pad)
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


def causal_attention_mask(length: int, device: torch.device) -> torch.Tensor:
    """Compact shared causal mask (``True`` entries are blocked)."""
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


# ---------------------------------------------------------------------------
# Rotary position embeddings (RoPE).  Added 2026-09 for the length-
# generalization reruns: every learned absolute position table in the paper's
# models can be swapped for RoPE via ``pos_kind="rope"``.  Positions are
# supplied per token (``pos_ids`` [B, L] or [L]); when omitted they default to
# 0..L-1.  Attention masks keep the codebase convention (bool, True = blocked,
# shape [B*H, S, S] or [S, S]).
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for rotate-half RoPE on a head dimension."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE needs an even head dimension")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pos_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pos_ids [..., L] (long) -> cos, sin [..., L, head_dim] (float32)."""
        freqs = pos_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k [B, H, L, D]; cos, sin [B, L, D] or [L, D] -> rotated q, k."""
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    dtype = q.dtype
    qf, kf = q.float(), k.float()
    q_out = qf * cos + _rotate_half(qf) * sin
    k_out = kf * cos + _rotate_half(kf) * sin
    return q_out.to(dtype), k_out.to(dtype)


def _blocked_mask_to_sdpa(mask: torch.Tensor | None, batch: int,
                          n_heads: int) -> torch.Tensor | None:
    """Codebase bool mask (True = blocked, [B*H, S, S] or [S, S]) ->
    SDPA bool mask (True = attend, [B, H, S, S] or [S, S])."""
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise ValueError("RoPE encoder expects a bool attention mask")
    allowed = ~mask
    if allowed.dim() == 2:
        return allowed
    if allowed.dim() == 3:
        S = allowed.shape[-1]
        if allowed.shape[0] == batch * n_heads:
            return allowed.view(batch, n_heads, S, S)
        if allowed.shape[0] == batch:
            return allowed.view(batch, 1, S, S)
    raise ValueError(f"unsupported attention mask shape {tuple(mask.shape)}")


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 rope_base: float = 10000.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, rope_base)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor,
                sdpa_mask: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, L, Dh]
        cos, sin = self.rope(pos_ids)
        q, k = apply_rope(q, k, cos, sin)
        drop = self.dropout if self.training else 0.0
        # CUDA SDPA kernels cap the batch grid at 65535 rows; the JEPA
        # counterfactual predictor batches can exceed that, so chunk.
        chunk = 32768
        if B <= chunk:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=drop,
                is_causal=sdpa_mask is None,
            )
        else:
            outs = []
            for i in range(0, B, chunk):
                m = sdpa_mask
                if m is not None and m.dim() == 4 and m.shape[0] == B:
                    m = m[i:i + chunk]
                outs.append(F.scaled_dot_product_attention(
                    q[i:i + chunk], k[i:i + chunk], v[i:i + chunk],
                    attn_mask=m, dropout_p=drop, is_causal=sdpa_mask is None,
                ))
            out = torch.cat(outs, 0)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class RoPETransformerBlock(nn.Module):
    """Pre-norm transformer block with rotary self-attention."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: float = 4,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RoPESelfAttention(d_model, n_heads, dropout, rope_base)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, int(round(d_model * ff_mult)))
        self.linear2 = nn.Linear(int(round(d_model * ff_mult)), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor,
                sdpa_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), pos_ids, sdpa_mask))
        x = x + self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
        return x


def _default_pos_ids(x: torch.Tensor, pos_ids: torch.Tensor | None) -> torch.Tensor:
    if pos_ids is None:
        return torch.arange(x.shape[1], device=x.device)
    return pos_ids


class RoPETransformerEncoder(nn.Module):
    """Drop-in for ``nn.TransformerEncoder`` (causal/masked use) with RoPE.

    ``forward(x, mask=None, src_key_padding_mask=None, pos_ids=None)``:
    ``mask`` follows the codebase convention (bool, True = blocked); with no
    mask the stack is plainly causal.  ``src_key_padding_mask`` (True = pad)
    is folded into the mask.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: float = 4,
                 n_layers: int = 1, dropout: float = 0.0,
                 rope_base: float = 10000.0, bidirectional: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.bidirectional = bool(bidirectional)
        self.layers = nn.ModuleList([
            RoPETransformerBlock(d_model, n_heads, ff_mult, dropout, rope_base)
            for _ in range(n_layers)
        ])

    def _sdpa_mask(self, x, mask, src_key_padding_mask):
        B, L, _ = x.shape
        sdpa = _blocked_mask_to_sdpa(mask, B, self.n_heads)
        if sdpa is None and self.bidirectional:
            sdpa = torch.ones(L, L, dtype=torch.bool, device=x.device)
        if src_key_padding_mask is not None:
            keep = ~src_key_padding_mask.bool()  # [B, L] True = valid key
            if sdpa is None:
                causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
                sdpa = causal[None, None] & keep[:, None, None, :]
            elif sdpa.dim() == 2:
                sdpa = sdpa[None, None] & keep[:, None, None, :]
            else:
                sdpa = sdpa & keep[:, None, None, :]
            dead = ~sdpa.any(-1, keepdim=True)
            sdpa = sdpa.clone()
            sdpa[..., 0:1] |= dead
        return sdpa

    def forward(self, x, mask=None, src_key_padding_mask=None, pos_ids=None):
        pos_ids = _default_pos_ids(x, pos_ids)
        sdpa = self._sdpa_mask(x, mask, src_key_padding_mask)
        for layer in self.layers:
            x = layer(x, pos_ids, sdpa)
        return x


class LoopedRoPEEncoder(RoPETransformerEncoder):
    """Weight-shared RoPE block with the LoopedTransformerEncoder schedule."""

    def __init__(self, d_model, n_heads, ff_mult, dropout=0.0,
                 train_loop_mean=4.0, train_loop_min=1, train_loop_max=8,
                 eval_loops=4, train_loop_distribution="shifted_poisson",
                 train_loop_sigma=0.5, rope_base=10000.0):
        super().__init__(d_model, n_heads, ff_mult, 1, dropout, rope_base)
        if dropout != 0.0:
            raise ValueError("looped reasoning baselines require dropout=0")
        self.train_loop_mean = float(train_loop_mean)
        self.train_loop_min = int(train_loop_min)
        self.train_loop_max = int(train_loop_max)
        self.eval_loops = int(eval_loops)
        self.train_loop_distribution = train_loop_distribution
        self.train_loop_sigma = float(train_loop_sigma)
        self.last_num_loops = self.eval_loops
        self.block = self.layers[0]

    sample_num_loops = LoopedTransformerEncoder.sample_num_loops

    def forward(self, x, mask=None, src_key_padding_mask=None, num_loops=None,
                pos_ids=None):
        loops = (
            self.sample_num_loops() if self.training else self.eval_loops
        ) if num_loops is None else int(num_loops)
        if loops < 1:
            raise ValueError("num_loops must be positive")
        self.last_num_loops = loops
        pos_ids = _default_pos_ids(x, pos_ids)
        sdpa = self._sdpa_mask(x, mask, src_key_padding_mask)
        for _ in range(loops):
            x = self.block(x, pos_ids, sdpa)
        return x


class RoPEDecoderBlock(nn.Module):
    """Pre-norm decoder block: rotary causal self-attention + cross-attention
    over a memory (no positions on the memory), used by the sentence LM's
    per-sentence token decoder."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: float = 4,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RoPESelfAttention(d_model, n_heads, dropout, rope_base)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, int(round(d_model * ff_mult)))
        self.linear2 = nn.Linear(int(round(d_model * ff_mult)), d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, pos_ids, sdpa_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), pos_ids, sdpa_mask))
        x = x + self.dropout(self.cross_attn(
            self.norm2(x), memory, memory, need_weights=False)[0])
        x = x + self.dropout(self.linear2(self.dropout(F.gelu(self.linear1(self.norm3(x))))))
        return x


class RoPETransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, n_layers=2, dropout=0.0,
                 rope_base=10000.0):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEDecoderBlock(d_model, n_heads, ff_mult, dropout, rope_base)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, pos_ids=None):
        pos_ids = _default_pos_ids(tgt, pos_ids)
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, pos_ids, None)  # None -> is_causal
        return self.norm(x)
