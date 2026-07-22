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


@lru_cache(maxsize=16)
def _compiled_flex_attention(capacity: int):
    """One static fused FlexAttention graph per power-of-two token bucket."""
    from torch.nn.attention.flex_attention import flex_attention
    # PyTorch 2.5 cannot lower this FlexAttention kernel with a symbolic token
    # dimension on Ada. Power-of-two token buckets bound slack below 2x while
    # keeping the number of compiled kernels logarithmic.
    return torch.compile(flex_attention, dynamic=False)


def _pad_flex_qkv(q, k, v, head_dim: int):
    """Pad only the kernel width; preserve the model's attention geometry."""
    kernel_dim = 1 << (int(head_dim) - 1).bit_length()
    if kernel_dim == head_dim:
        return q, k, v
    amount = kernel_dim - head_dim
    return tuple(F.pad(part, (0, amount)) for part in (q, k, v))


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
    ) -> torch.Tensor:
        """Self-attention over a flat, boundary-separated token stream.

        ``query`` contains no padding. ``cu_seqlens`` identifies independent
        examples; attention is therefore exactly block diagonal and can never
        leak information between packed training items.
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
                causal=False,
            )
            self.last_backend = "flash_attn_4_packed"
        elif backend == "flash_attn_2":
            _, fa2 = _external_flash_functions()
            if fa2 is None:
                raise RuntimeError("FlashAttention-2 was requested but is not installed")
            attended = fa2(
                qkv, cu_seqlens, max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )
            self.last_backend = "flash_attn_2_packed"
        elif query.is_cuda:
            if self.dropout:
                raise ValueError("FlexAttention packed path requires dropout=0")
            if block_mask is None:
                raise ValueError("packed torch attention requires a block mask")
            q, k, v = (part.transpose(0, 1).unsqueeze(0) for part in qkv.unbind(1))
            # PyTorch 2.5 FlexAttention requires a power-of-two head width.
            # Padding projected features is algebraically neutral provided we
            # retain the original attention scale and slice the values back.
            q, k, v = _pad_flex_qkv(q, k, v, self.head_dim)
            attended = _compiled_flex_attention(query.shape[0])(
                q, k, v, block_mask=block_mask,
                scale=self.head_dim ** -0.5,
            )[..., :self.head_dim]
            attended = attended.squeeze(0).transpose(0, 1)
            self.last_backend = "torch_flex_packed"
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
                    is_causal=False,
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
) -> torch.Tensor:
    """Run a pre-norm encoder with no padded activations between layers.

    The returned tensor is dense only at the model boundary for compatibility
    with token-aligned losses and edit routing. All attention and feed-forward
    blocks operate on ``sum(valid)`` tokens.
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
    block_mask = None
    first_attention = encoder.layers[0].self_attn if encoder.layers else None
    if flat.is_cuda and isinstance(first_attention, FlashMultiheadAttention):
        if first_attention._select_backend(flat) in {"torch", "torch_flash"}:
            from torch.nn.attention.flex_attention import create_block_mask
            capacity = max(128, 1 << (actual_tokens - 1).bit_length())
            sequence_ids = torch.empty(
                (capacity,), dtype=torch.int32, device=flat.device
            )
            sequence_ids[:actual_tokens] = torch.repeat_interleave(
                torch.arange(len(lengths), device=flat.device, dtype=torch.int32),
                lengths.long(),
            )
            if capacity > actual_tokens:
                # Every slack token is its own isolated dummy sequence. It can
                # neither read real examples nor become a key for them.
                sequence_ids[actual_tokens:] = torch.arange(
                    len(lengths), len(lengths) + capacity - actual_tokens,
                    device=flat.device, dtype=torch.int32,
                )
                flat = torch.cat([
                    flat,
                    flat.new_zeros(capacity - actual_tokens, flat.shape[-1]),
                ])

            def same_sequence(batch, head, query_index, key_index):
                return sequence_ids[query_index] == sequence_ids[key_index]

            block_mask = create_block_mask(
                same_sequence, B=1, H=None, Q_LEN=capacity, KV_LEN=capacity,
                device=flat.device, _compile=False,
            )
    for layer in encoder.layers:
        if not layer.norm_first:
            raise ValueError("packed encoder currently requires norm_first=True")
        if not isinstance(layer.self_attn, FlashMultiheadAttention):
            raise ValueError(
                "packed encoder requires a fused attention backend, not 'torch'"
            )
        attended = layer.self_attn.forward_packed(
            layer.norm1(flat), cu, maximum, block_mask=block_mask
        )
        flat = flat + layer.dropout1(attended)
        flat = flat + layer._ff_block(layer.norm2(flat))
    if encoder.norm is not None:
        flat = encoder.norm(flat)
    output = src.new_zeros(src.shape)
    output[valid] = flat[:actual_tokens]
    return output


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
