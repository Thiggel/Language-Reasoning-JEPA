"""Two-level edit JEPA with token dynamics and sentence subgoals.

Unlike the historical edit model, token representations are contextualized in
one bidirectional pass over ``[prompt | complete buffer]``.  Learned attention
pooling then maps buffer tokens to sentences and a second bidirectional encoder
constructs a distinct sentence representation space.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import MacroActionModel
from textjepa.models.delta_decoder import ObservedActionDecoder
from textjepa.models.ema import EMATeacher
from textjepa.models.layers import encoder_stack, packed_encoder_forward
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.predictor import TokenAlignedEditPredictor


def _masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(-2) / weight.sum(-2).clamp_min(1)


class HierarchicalBufferEncoder(nn.Module):
    """Whole-sequence token encoder followed by attention-pooled sentences."""

    def __init__(self, vocab_size: int, pad_id: int, d_model: int = 256,
                 token_layers: int = 2, sentence_layers: int = 2,
                 n_heads: int = 8, ff_mult: int = 4,
                 max_sequence_len: int = 1024, max_sentences: int = 64,
                 dropout: float = 0.0, pooling: str = "attention",
                 attention_backend: str = "torch",
                 sequence_packing: bool = False):
        super().__init__()
        if pooling not in {"attention", "mean"}:
            raise ValueError(f"unknown sentence pooling: {pooling}")
        self.pooling = pooling
        self.pad_id = int(pad_id)
        self.sequence_packing = bool(sequence_packing)
        self.max_sequence_len = int(max_sequence_len)
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.token_pos = nn.Parameter(torch.zeros(1, max_sequence_len, d_model))
        self.segment = nn.Parameter(torch.zeros(2, d_model))
        self.sentence_start = nn.Parameter(torch.zeros(d_model))
        self.token_encoder = encoder_stack(
            d_model, token_layers, n_heads, ff_mult, dropout, attention_backend
        )
        self.token_norm = nn.LayerNorm(d_model)
        # A learned scalar attention score, normalized independently inside
        # every sentence.  This is intentionally not masked mean pooling.
        self.pool_score = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.Tanh(), nn.Linear(d_model // 2, 1, bias=False),
        )
        self.sentence_pos = nn.Parameter(torch.zeros(1, max_sentences, d_model))
        self.sentence_encoder = encoder_stack(
            d_model, sentence_layers, n_heads, ff_mult, dropout,
            attention_backend,
        )
        self.sentence_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.token_pos, std=0.02)
        nn.init.normal_(self.segment, std=0.02)
        nn.init.normal_(self.sentence_start, std=0.02)
        nn.init.normal_(self.sentence_pos, std=0.02)

    @staticmethod
    def _pack(prompt: torch.Tensor, buffer: torch.Tensor, pad_id: int):
        """Pack valid tokens and retain -1(prompt)/sentence buffer labels."""
        n, c, length = buffer.shape
        prompt_flat = prompt.reshape(n, -1)
        buffer_flat = buffer.reshape(n, -1)
        values = torch.cat([prompt_flat, buffer_flat], dim=1)
        keep = values.ne(pad_id)
        labels = torch.cat([
            prompt_flat.new_full(prompt_flat.shape, -1),
            torch.arange(c, device=buffer.device, dtype=buffer.dtype)
            .view(1, c, 1).expand(n, c, length).reshape(n, -1),
        ], dim=1)
        counts = keep.sum(1)
        width = max(int(counts.max().item()), 1)
        tokens = prompt.new_full((n, width), pad_id)
        sentence_ids = prompt.new_full((n, width), -2)
        destination = keep.long().cumsum(1) - 1
        row = torch.arange(n, device=prompt.device).unsqueeze(1).expand_as(keep)
        tokens[row[keep], destination[keep]] = values[keep]
        sentence_ids[row[keep], destination[keep]] = labels[keep]
        valid = torch.arange(width, device=prompt.device).unsqueeze(0) < counts[:, None]
        return tokens, valid, sentence_ids

    def contextual_tokens(self, prompt: torch.Tensor, buffer: torch.Tensor):
        tokens, valid, sentence_ids = self._pack(prompt, buffer, self.pad_id)
        if tokens.shape[1] > self.max_sequence_len:
            raise ValueError(
                f"packed sequence length {tokens.shape[1]} exceeds "
                f"max_sequence_len={self.max_sequence_len}"
            )
        segment = sentence_ids.ge(0).long().clamp(0, 1)
        h = self.tok(tokens) + self.token_pos[:, :tokens.shape[1]]
        h = h + self.segment[segment]
        previous = torch.nn.functional.pad(
            sentence_ids[:, :-1], (1, 0), value=-1
        )
        sentence_start = sentence_ids.ge(0) & sentence_ids.ne(previous)
        h = h + sentence_start.unsqueeze(-1) * self.sentence_start
        key_pad = ~valid
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        encoded = (
            packed_encoder_forward(self.token_encoder, h, valid)
            if self.sequence_packing else
            self.token_encoder(h, src_key_padding_mask=key_pad)
        )
        h = self.token_norm(encoded)
        buffer_valid = valid & sentence_ids.ge(0)
        widths = buffer_valid.sum(-1)
        width = max(int(widths.max().item()), 1)
        out = h.new_zeros(h.shape[0], width, h.shape[-1])
        out_ids = sentence_ids.new_full((h.shape[0], width), -1)
        destination = buffer_valid.long().cumsum(1) - 1
        row = torch.arange(h.shape[0], device=h.device).unsqueeze(1).expand_as(
            buffer_valid
        )
        out[row[buffer_valid], destination[buffer_valid]] = h[buffer_valid]
        out_ids[row[buffer_valid], destination[buffer_valid]] = sentence_ids[
            buffer_valid
        ]
        out_mask = (
            torch.arange(width, device=h.device).unsqueeze(0) < widths[:, None]
        )
        return out, out_mask, out_ids

    def pool_sentences(self, token_states: torch.Tensor,
                       token_mask: torch.Tensor, sentence_ids: torch.Tensor,
                       n_sentences: int):
        if n_sentences > self.sentence_pos.shape[1]:
            raise ValueError("too many sentences for configured sentence positions")
        n, width, dim = token_states.shape
        raw_score = self.pool_score(token_states).squeeze(-1)
        members = token_mask & sentence_ids.ge(0) & sentence_ids.lt(n_sentences)
        groups = sentence_ids.clamp(0, n_sentences - 1)
        if self.pooling == "attention":
            maxima = raw_score.new_full((n, n_sentences), -torch.inf)
            maxima.scatter_reduce_(
                1, groups, raw_score.masked_fill(~members, -torch.inf),
                reduce="amax", include_self=True,
            )
            centered = (raw_score - maxima.gather(1, groups)).masked_fill(
                ~members, -torch.inf
            )
            weight = centered.exp()
        else:
            weight = members.to(raw_score.dtype)
        denominator = raw_score.new_zeros(n, n_sentences)
        # CUDA autocast may promote ``exp`` to FP32 while the linear score and
        # its denominator remain BF16. In-place scatter requires an exact dtype
        # match; returning to the score dtype also keeps pooled activations on
        # the model's intended precision path.
        weight = weight.to(denominator.dtype)
        denominator.scatter_add_(1, groups, weight)
        attention = weight / denominator.gather(1, groups).clamp_min(1)
        attention = attention * members.to(attention.dtype)
        pooled = token_states.new_zeros(n, n_sentences, dim)
        pooled.scatter_add_(
            1, groups.unsqueeze(-1).expand(-1, -1, dim),
            token_states * attention.unsqueeze(-1),
        )
        sentence_mask = denominator.gt(0)
        key_pad = ~sentence_mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        sentence_input = pooled + self.sentence_pos[:, :n_sentences]
        encoded = (
            packed_encoder_forward(
                self.sentence_encoder, sentence_input, sentence_mask
            )
            if self.sequence_packing else
            self.sentence_encoder(
                sentence_input, src_key_padding_mask=key_pad,
            )
        )
        encoded = self.sentence_norm(encoded)
        encoded = encoded * sentence_mask.unsqueeze(-1)
        return encoded, sentence_mask, attention

    def forward(self, prompt: torch.Tensor, buffer: torch.Tensor):
        """Inputs [N,P,L], [N,C,L]; return both representation levels."""
        tokens, token_mask, ids = self.contextual_tokens(prompt, buffer)
        sentences, sentence_mask, attention = self.pool_sentences(
            tokens, token_mask, ids, buffer.shape[1]
        )
        return tokens, token_mask, ids, sentences, sentence_mask, attention


class SentenceEditPredictor(nn.Module):
    """Bidirectional sentence transition with local or global action injection."""

    def __init__(self, d_model: int, d_action: int, n_layers: int = 2,
                 n_heads: int = 8, correction: bool = False,
                 attention_backend: str = "torch",
                 sequence_packing: bool = False):
        super().__init__()
        self.correction = correction
        self.sequence_packing = bool(sequence_packing)
        self.action = nn.Linear(d_action, d_model)
        self.current = nn.Linear(d_model, d_model) if correction else None
        self.blocks = encoder_stack(
            d_model, n_layers, n_heads, 4, 0.0, attention_backend
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, base: torch.Tensor, mask: torch.Tensor,
                action: torch.Tensor, affected: torch.Tensor | None,
                current: torch.Tensor | None = None):
        h = base
        if self.correction:
            if current is None:
                raise ValueError("correction predictor requires current sentences")
            h = h + self.current(current)
        cond = self.action(action)
        if affected is None:
            h = h + cond.unsqueeze(1)
        else:
            local = torch.zeros_like(h)
            row = torch.arange(len(h), device=h.device)
            index = affected.clamp(0, h.shape[1] - 1)
            local[row, index] = cond.to(local.dtype)
            h = h + local
        key_pad = ~mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        encoded = (
            packed_encoder_forward(self.blocks, h, mask)
            if self.sequence_packing else
            self.blocks(h, src_key_padding_mask=key_pad)
        )
        delta = self.out(encoded)
        return self.norm(base + delta) * mask.unsqueeze(-1)


class PrimitiveEditActionEncoder(nn.Module):
    """Pointer-relative primitive action code used by sentence-only control."""

    def __init__(self, d_model: int, d_action: int):
        super().__init__()
        self.op = nn.Embedding(3, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, d_model),
            nn.GELU(), nn.Linear(d_model, d_action),
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor,
                operations: torch.Tensor, positions: torch.Tensor,
                content: torch.Tensor):
        left, right = TokenAlignedEditPredictor._gather_context(
            states, mask, positions
        )
        return self.net(torch.cat([
            self.op(operations.clamp(0, 2)), left, right, content
        ], -1))

    def encode_candidates(
        self, states: torch.Tensor, mask: torch.Tensor,
        operations: torch.Tensor, positions: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        """Encode K pointer-relative actions without copying state sequences."""
        n = states.shape[0]
        lengths = mask.sum(-1).long()
        left_index = torch.minimum(
            (positions - 1).clamp_min(0),
            (lengths - 1).clamp_min(0).unsqueeze(-1),
        )
        right_index = torch.minimum(
            positions.clamp_min(0),
            (lengths - 1).clamp_min(0).unsqueeze(-1),
        )
        row = torch.arange(n, device=states.device).unsqueeze(-1)
        active = lengths.gt(0).view(n, 1, 1)
        left = states[row, left_index] * active
        right = states[row, right_index] * active
        return self.net(torch.cat([
            self.op(operations.clamp(0, 2)), left, right, content,
        ], -1))


class SentencePatternActionEncoder(nn.Module):
    """Encode ``[blank ... replacement ... blank]`` without coordinates.

    The action has the length of the affected sentence.  A learned blank is
    placed at every unchanged slot and the replacement-token embedding at the
    selected slot.  A small Transformer and learned attention pool compress
    the complete structural pattern into ``d_action`` dimensions.
    """

    def __init__(self, d_model: int, d_action: int, n_heads: int = 8,
                 max_sentence_len: int = 256,
                 attention_backend: str = "torch"):
        super().__init__()
        self.blank = nn.Parameter(torch.zeros(d_model))
        self.relative = nn.Parameter(torch.zeros(1, max_sentence_len, d_model))
        self.blocks = encoder_stack(
            d_model, 1, n_heads, 2, 0.0, attention_backend
        )
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.out = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_action),
        )
        nn.init.normal_(self.blank, std=0.02)
        nn.init.normal_(self.relative, std=0.02)

    def forward(self, states: torch.Tensor, mask: torch.Tensor,
                sentence_ids: torch.Tensor, operations: torch.Tensor,
                positions: torch.Tensor, content: torch.Tensor):
        del states, operations  # routing and content are the complete action
        lengths = mask.sum(1)
        selected_position = positions.clamp_min(0).minimum(
            (lengths - 1).clamp_min(0)
        )
        row = torch.arange(len(mask), device=mask.device)
        selected_sentence = sentence_ids[row, selected_position].clamp_min(0)
        members = mask & sentence_ids.eq(selected_sentence[:, None])
        member_count = members.sum(1).clamp_min(1)
        before_or_at = (
            torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
            <= selected_position[:, None]
        )
        selected_offset = (members & before_or_at).sum(1).sub(1).clamp_min(0)
        width = int(member_count.max().item())
        if width > self.relative.shape[1]:
            raise ValueError(
                f"sentence action length {width} exceeds {self.relative.shape[1]}"
            )
        pattern = self.blank.view(1, 1, -1).expand(len(mask), width, -1).clone()
        valid = (
            torch.arange(width, device=mask.device).unsqueeze(0)
            < member_count[:, None]
        )
        pattern[row, selected_offset] = content.to(pattern.dtype)
        pattern = pattern + self.relative[:, :width]
        h = self.blocks(pattern, src_key_padding_mask=~valid)
        logits = self.score(h).squeeze(-1).masked_fill(~valid, -torch.inf)
        pooled = (h * logits.softmax(-1).unsqueeze(-1)).sum(1)
        return self.out(pooled)


class TokenReplacementPrior(nn.Module):
    """Factorized deployment prior over ``position`` then ``token``.

    The prior consumes only the current online state and prompt.  In
    particular, neither the clean buffer nor an EMA goal is an input.  The
    optional stop-gradient is the clean ablation between a read-only policy
    head and a policy loss that is allowed to shape the representation.
    """

    def __init__(self, d_model: int, vocab_size: int, detach_state: bool,
                 predict_position: bool = True):
        super().__init__()
        self.detach_state = bool(detach_state)
        self.predict_position = bool(predict_position)
        self.position = nn.Sequential(
            nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model),
            nn.GELU(), nn.Linear(d_model, 1),
        ) if self.predict_position else None
        self.content = nn.Sequential(
            nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model),
            nn.GELU(), nn.Linear(d_model, vocab_size),
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor,
                prompt: torch.Tensor, positions: torch.Tensor):
        if self.detach_state:
            states, prompt = states.detach(), prompt.detach()
        pooled = _masked_pool(states, mask)
        shared = torch.cat([
            states, pooled.unsqueeze(1).expand_as(states),
            prompt.unsqueeze(1).expand_as(states),
        ], -1)
        position_logits = None
        if self.position is not None:
            position_logits = self.position(shared).squeeze(-1)
            position_logits = position_logits.masked_fill(~mask, -torch.inf)
            row = torch.arange(len(states), device=states.device)
            selected = states[
                row, positions.clamp(0, states.shape[1] - 1)
            ]
            content_logits = self.content(
                torch.cat([selected, pooled, prompt], -1)
            )
        else:
            # One content distribution per structurally grounded token slot.
            # Search enumerates slots; no arbitrary expert order is learned.
            content_logits = self.content(shared)
        return position_logits, content_logits


class MacroOptionDecoder(nn.Module):
    """Decode a macro code into a closed-loop primitive replacement policy.

    This is an option policy, not a language decoder.  It is called again
    after every mechanically executed replacement, so position logits always
    refer to the current token state rather than an obsolete open-loop index.
    """

    def __init__(self, d_model: int, d_macro: int, vocab_size: int,
                 span: int, detach_inputs: bool = True):
        super().__init__()
        self.detach_inputs = bool(detach_inputs)
        self.macro = nn.Linear(d_macro, d_model)
        self.step = nn.Embedding(span, d_model)
        self.position = nn.Sequential(
            nn.LayerNorm(5 * d_model), nn.Linear(5 * d_model, d_model),
            nn.GELU(), nn.Linear(d_model, 1),
        )
        self.content = nn.Sequential(
            nn.LayerNorm(5 * d_model), nn.Linear(5 * d_model, d_model),
            nn.GELU(), nn.Linear(d_model, vocab_size),
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor,
                prompt: torch.Tensor, macro: torch.Tensor,
                step: torch.Tensor, positions: torch.Tensor):
        if self.detach_inputs:
            states, prompt, macro = (
                states.detach(), prompt.detach(), macro.detach()
            )
        pooled = _masked_pool(states, mask)
        macro_emb = self.macro(macro)
        step_emb = self.step(step.clamp(0, self.step.num_embeddings - 1))
        common = [pooled, prompt, macro_emb, step_emb]
        position_input = torch.cat([
            states, *[
                value.unsqueeze(1).expand_as(states) for value in common
            ],
        ], -1)
        position_logits = self.position(position_input).squeeze(-1)
        position_logits = position_logits.masked_fill(~mask, -torch.inf)
        row = torch.arange(len(states), device=states.device)
        selected = states[
            row, positions.clamp(0, states.shape[1] - 1)
        ]
        content_logits = self.content(torch.cat([selected, *common], -1))
        return position_logits, content_logits


class MultiscaleEditJEPA(nn.Module):
    """Four controlled variants of token/sentence edit dynamics.

    ``token``: token transition only.
    ``sentence``: primitive sentence transition only.
    ``token_sentence``: token transition plus sentence correction.
    ``sentence_macro``: direct sentence dynamics plus K-action sentence subgoals.
    ``token_sentence_macro``: token correction plus K-action sentence subgoals.
    """

    VALID_VARIANTS = {"token", "sentence", "sentence_macro", "token_sentence",
                      "token_sentence_macro"}

    def __init__(self, vocab_size: int, pad_id: int, variant: str,
                 d_model: int = 256, d_action: int = 16, d_macro: int = 8,
                 macro_k: int = 4, token_layers: int = 2,
                 sentence_layers: int = 2, predictor_layers: int = 2,
                 n_heads: int = 8, ff_mult: int = 4,
                 max_sequence_len: int = 1024, max_sentences: int = 64,
                 token_relative_radius: int = 32,
                 observed_action_ldad: bool = False,
                 ldad_max_len: int = 12, dropout: float = 0.0,
                 max_transitions_per_forward: int = 8,
                 sentence_pooling: str = "attention",
                 macro_prior_detach_state: bool = True,
                 base_prior: bool = False,
                 base_prior_detach_state: bool = True,
                 macro_decoder: bool = False,
                 macro_decoder_detach_inputs: bool = True,
                 sentence_action_kind: str = "context",
                 direct_content_scaffold: bool = True,
                 base_prior_predict_position: bool = True,
                 base_q_hidden: int | None = None,
                 base_q_primitive_action: bool = False,
                 attention_backend: str = "torch",
                 sequence_packing: bool = False,
                 efficient_pair_encoding: bool = False,
                 target_encoder_mode: str = "ema"):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown multiscale edit variant: {variant}")
        if dropout != 0:
            raise ValueError("multiscale edit JEPA requires dropout=0")
        if sentence_action_kind not in {"context", "blank_pattern"}:
            raise ValueError(f"unknown sentence_action_kind: {sentence_action_kind}")
        if target_encoder_mode not in {
            "ema", "shared_stopgrad", "shared_symmetric"
        }:
            raise ValueError(
                f"unknown target_encoder_mode: {target_encoder_mode}"
            )
        self.variant = variant
        self.use_token_loss = variant not in {"sentence", "sentence_macro"}
        self.use_sentence = variant != "token"
        self.use_macro = variant in {"sentence_macro", "token_sentence_macro"}
        self.macro_k = int(macro_k)
        self.macro_prior_detach_state = bool(macro_prior_detach_state)
        self.max_transitions_per_forward = max(
            0, int(max_transitions_per_forward)
        )
        self.efficient_pair_encoding = bool(efficient_pair_encoding)
        self.target_encoder_mode = target_encoder_mode
        self.detach_targets = target_encoder_mode != "shared_symmetric"
        self.encoder = HierarchicalBufferEncoder(
            vocab_size, pad_id, d_model, token_layers, sentence_layers,
            n_heads, ff_mult, max_sequence_len, max_sentences, dropout,
            sentence_pooling, attention_backend, sequence_packing,
        )
        self.teacher = (
            EMATeacher(self.encoder) if target_encoder_mode == "ema" else None
        )
        self.token_pred = None if variant in {"sentence", "sentence_macro"} else TokenAlignedEditPredictor(
            d_model, d_action, predictor_layers, n_heads,
            relative_radius=token_relative_radius,
            direct_content_scaffold=direct_content_scaffold,
            attention_backend=attention_backend,
            sequence_packing=sequence_packing,
            replacement_only_fast_path=efficient_pair_encoding,
        )
        self.sentence_action = None
        if variant in {"sentence", "sentence_macro"}:
            self.sentence_action = (
                SentencePatternActionEncoder(
                    d_model, d_action, n_heads,
                    attention_backend=attention_backend,
                )
                if sentence_action_kind == "blank_pattern"
                else PrimitiveEditActionEncoder(d_model, d_action)
            )
        self.sentence_pred = None if not self.use_sentence else SentenceEditPredictor(
            d_model, d_action, predictor_layers, n_heads,
            correction=variant not in {"sentence", "sentence_macro"},
            attention_backend=attention_backend,
            sequence_packing=sequence_packing,
        )
        self.macro_model = None
        self.macro_pred = None
        if self.use_macro:
            if self.macro_k < 2:
                raise ValueError("macro hierarchy requires macro_k >= 2")
            self.macro_model = MacroActionModel(
                d_action, d_model, d_macro, self.macro_k,
                kind="concat", concat_width=min(d_action, 8),
            )
            self.macro_pred = SentenceEditPredictor(
                d_model, d_macro, predictor_layers, n_heads,
                attention_backend=attention_backend,
                sequence_packing=sequence_packing,
            )
        self.macro_decoder = None
        if macro_decoder:
            if not self.use_macro:
                raise ValueError("macro_decoder requires a macro model variant")
            self.macro_decoder = MacroOptionDecoder(
                d_model, d_macro, vocab_size, self.macro_k,
                macro_decoder_detach_inputs,
            )
        self.ldad = ObservedActionDecoder(
            d_model, vocab_size, ldad_max_len,
            n_layers=predictor_layers, n_heads=n_heads,
        ) if observed_action_ldad and self.use_sentence else None
        self.base_prior = TokenReplacementPrior(
            d_model, vocab_size, base_prior_detach_state,
            base_prior_predict_position,
        ) if base_prior else None
        self.base_q_primitive_action = bool(base_q_primitive_action)
        self.base_q_action = (
            PrimitiveEditActionEncoder(d_model, d_action)
            if self.base_q_primitive_action else None
        )
        q_hidden = d_model if base_q_hidden is None else int(base_q_hidden)
        if q_hidden < 1:
            raise ValueError("base_q_hidden must be positive")
        self.base_q_head = nn.Sequential(
            nn.LayerNorm(d_model + d_action),
            nn.Linear(d_model + d_action, q_hidden), nn.GELU(),
            nn.Linear(q_hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def replacement_prior(self, states: torch.Tensor, mask: torch.Tensor,
                          prompt: torch.Tensor, positions: torch.Tensor):
        if self.base_prior is None:
            raise RuntimeError("checkpoint has no base token-action prior")
        return self.base_prior(states, mask, prompt, positions)

    def action_value(self, state: torch.Tensor,
                     action: torch.Tensor) -> torch.Tensor:
        """Deployment-time V(s,a); the privileged goal is never an input."""
        return self.base_q_head(torch.cat([state, action], -1)).squeeze(-1)

    @torch.no_grad()
    def update_teachers(self, momentum: float):
        if self.teacher is not None:
            self.teacher.update(self.encoder, momentum)

    @staticmethod
    def affected_sentences(ids: torch.Tensor, mask: torch.Tensor,
                           operations: torch.Tensor, positions: torch.Tensor):
        """Map a pointer/gap to a sentence without a mutable absolute register."""
        del operations  # every pointer/gap is owned by the token on its right
        lengths = mask.sum(1)
        pos = positions.clamp_min(0).minimum((lengths - 1).clamp_min(0))
        row = torch.arange(len(ids), device=ids.device)
        result = ids[row, pos].clamp_min(0)
        return torch.where(lengths.gt(0), result, torch.zeros_like(result))

    @staticmethod
    def transition_sentence_ids(ids: torch.Tensor, mask: torch.Tensor,
                                operations: torch.Tensor,
                                positions: torch.Tensor):
        """Apply the same structural edit as the token predictor to labels."""
        if operations.eq(2).all():
            return (
                ids.clone(), mask.clone(),
                MultiscaleEditJEPA.affected_sentences(
                    ids, mask, operations, positions
                ),
            )
        out = ids.new_full(ids.shape, -1)
        out_mask = torch.zeros_like(mask)
        affected = MultiscaleEditJEPA.affected_sentences(
            ids, mask, operations, positions
        )
        for row in range(len(ids)):
            length = int(mask[row].sum().item())
            current = ids[row, :length]
            pos = int(positions[row].item())
            op = int(operations[row].item())
            pos = min(max(pos, 0), length if op == 1 else max(length - 1, 0))
            if op == 0:
                edited = torch.cat([current[:pos], current[pos + 1:]])
            elif op == 1:
                label = affected[row:row + 1]
                edited = torch.cat([current[:pos], label, current[pos:]])
            else:
                edited = current
            count = min(len(edited), out.shape[1])
            out[row, :count] = edited[:count]
            out_mask[row, :count] = True
        return out, out_mask, affected

    def _encode_buffers(self, batch: dict, buffers: torch.Tensor,
                        teacher: bool = False):
        b, states, sentences, length = buffers.shape
        prompt = batch["prompt_tokens"].unsqueeze(1).expand(
            b, states, *batch["prompt_tokens"].shape[1:]
        )
        if teacher and self.teacher is None:
            raise RuntimeError("teacher encoding requested without an EMA teacher")
        module = self.teacher if teacher else self.encoder
        result = module(
            prompt.reshape(b * states, *prompt.shape[2:]),
            buffers.reshape(b * states, sentences, length),
        )
        token, token_mask, ids, sent, sent_mask, attention = result
        return (
            token.reshape(b, states, *token.shape[1:]),
            token_mask.reshape(b, states, -1),
            ids.reshape(b, states, -1),
            sent.reshape(b, states, sentences, -1),
            sent_mask.reshape(b, states, sentences),
            attention.reshape(b, states, -1),
        )

    def _encode_trajectory(self, batch: dict, teacher: bool = False):
        return self._encode_buffers(batch, batch["buffer_tokens"], teacher)

    def _encode_transition_pair(self, batch: dict):
        """Encode one current/next pair according to the target policy.

        EMA and shared-stop-gradient modes avoid storing target activations but
        still require a second forward encoding of the next state.  Symmetric
        mode batches both states into one encoder call and lets prediction loss
        gradients reach both sides, as required by heuristic-free regularizers.
        """
        if self.target_encoder_mode == "shared_symmetric":
            online = self._encode_buffers(batch, batch["buffer_tokens"][:, :2])
            return online, online
        current = self._encode_buffers(batch, batch["buffer_tokens"][:, :1])
        with torch.no_grad():
            target = self._encode_buffers(
                batch, batch["buffer_tokens"][:, 1:2],
                teacher=self.target_encoder_mode == "ema",
            )
        # Preserve the long-standing [current, next] output contract. The
        # second online slot is the detached EMA state because no enabled
        # transition objective consumes an online encoding of the next state.
        online = tuple(
            torch.cat([left, right.detach()], dim=1)
            for left, right in zip(current, target)
        )
        # Target index one is the next-state target. Index zero is unused.
        teacher = tuple(
            torch.cat([left.detach(), right], dim=1)
            for left, right in zip(current, target)
        )
        return online, teacher

    def _limit_trajectory(self, batch: dict) -> int:
        """Sample a contiguous exact-transition segment to bound O(T L^2).

        Iterative unmasking can have one full-buffer snapshot per token.  A
        full bidirectional encoder over every snapshot at once is neither
        needed for a stationary transition model nor computationally viable.
        The same contiguous slice is applied in-place to all step-aligned
        fields, so objectives and generic trainer metrics cannot drift out of
        alignment.  Macro windows remain genuinely consecutive.
        """
        total = batch["buffer_tokens"].shape[1] - 1
        keep = self.max_transitions_per_forward
        if not keep or total <= keep:
            return 0
        maximum_start = total - keep
        if self.training:
            start = int(torch.randint(
                maximum_start + 1, (), device=batch["buffer_tokens"].device
            ).item())
        else:
            start = maximum_start // 2
        batch["buffer_tokens"] = batch["buffer_tokens"][:, start:start + keep + 1]
        batch["buffer_mask"] = batch["buffer_mask"][:, start:start + keep + 1]
        if "goal_distance" in batch:
            batch["goal_distance"] = batch["goal_distance"][
                :, start:start + keep + 1
            ]
        excluded = {
            "prompt_tokens", "prompt_mask", "buffer_tokens", "buffer_mask",
            "goal_distance",
            "goal_buffer_tokens", "goal_buffer_mask", "answer", "n_necessary",
            "n_vars", "index",
        }
        for name, value in list(batch.items()):
            if (name not in excluded and torch.is_tensor(value)
                    and value.ndim >= 2 and value.shape[1] == total):
                batch[name] = value[:, start:start + keep]
        return start

    def forward(self, batch: dict) -> JEPAOutputs:
        transition_start = self._limit_trajectory(batch)
        use_pair_encoding = (
            self.efficient_pair_encoding
            and batch["buffer_tokens"].shape[1] == 2
            and batch["op"].shape[1] == 1
        )
        if use_pair_encoding:
            online, target = self._encode_transition_pair(batch)
            tokens, token_mask, ids, sentences, sentence_mask, attention = online
            (tgt_tokens, tgt_token_mask, _, tgt_sentences,
             tgt_sentence_mask, _) = target
        else:
            tokens, token_mask, ids, sentences, sentence_mask, attention = (
                self._encode_trajectory(batch)
            )
            if self.target_encoder_mode == "ema":
                with torch.no_grad():
                    (tgt_tokens, tgt_token_mask, _, tgt_sentences,
                     tgt_sentence_mask, _) = self._encode_trajectory(
                        batch, teacher=True
                    )
            else:
                # Every trajectory state is already online-encoded above.
                # Reuse it rather than performing an identical shared-encoder
                # pass. Detachment is applied at the output boundary below.
                tgt_tokens, tgt_token_mask = tokens, token_mask
                tgt_sentences, tgt_sentence_mask = sentences, sentence_mask
        b, states, width, dim = tokens.shape
        steps = states - 1
        op = batch["op"][:, :steps]
        pos = batch["edit_position"][:, :steps]
        content = self.encoder.tok(batch["edit_content_token"][:, :steps])
        prompt_mask = batch["prompt_tokens"].ne(self.encoder.pad_id)
        prompt_emb = _masked_pool(
            self.encoder.tok(batch["prompt_tokens"].reshape(b, -1)),
            prompt_mask.reshape(b, -1),
        )
        current = tokens[:, :-1].reshape(b * steps, width, dim)
        current_mask = token_mask[:, :-1].reshape(b * steps, width)
        if self.token_pred is None:
            if isinstance(self.sentence_action, SentencePatternActionEncoder):
                action = self.sentence_action(
                    current, current_mask, ids[:, :-1].reshape(b * steps, width),
                    op.reshape(-1), pos.reshape(-1), content.reshape(-1, dim),
                )
            else:
                action = self.sentence_action(
                    current, current_mask, op.reshape(-1), pos.reshape(-1),
                    content.reshape(-1, dim),
                )
        else:
            action = self.token_pred.encode_action(
                current, current_mask, op.reshape(-1), pos.reshape(-1),
                content.reshape(-1, dim),
            )
        action = action.reshape(b, steps, -1)
        q_action = (
            self.base_q_action(
                current, current_mask, op.reshape(-1), pos.reshape(-1),
                content.reshape(-1, dim),
            ).reshape(b, steps, -1)
            if self.base_q_action is not None else action
        )
        affected = self.affected_sentences(
            ids[:, :-1].reshape(b * steps, width), current_mask,
            op.reshape(-1), pos.reshape(-1),
        ).reshape(b, steps)

        if self.token_pred is not None:
            token_pred, predicted_mask = self.token_pred(
                current, current_mask, op.reshape(-1), pos.reshape(-1),
                content.reshape(-1, dim),
                prompt_emb[:, None].expand(b, steps, dim).reshape(-1, dim),
            )
            token_pred = token_pred.reshape(b, steps, width, dim)
            predicted_mask = predicted_mask.reshape(b, steps, width)
        else:
            # Explicit sentinel: sentence-only has no token transition path.
            token_pred, predicted_mask = None, token_mask[:, 1:]

        sentence_pred = None
        if self.use_sentence:
            if self.variant in {"sentence", "sentence_macro"}:
                base = sentences[:, :-1]
            else:
                if use_pair_encoding:
                    next_ids = ids[:, :-1].reshape(b * steps, width)
                    next_mask = current_mask
                else:
                    next_ids, next_mask, _ = self.transition_sentence_ids(
                        ids[:, :-1].reshape(b * steps, width), current_mask,
                        op.reshape(-1), pos.reshape(-1),
                    )
                # The lower prediction is re-encoded into the macro space;
                # no target state or target boundary enters this path.
                base, _, _ = self.encoder.pool_sentences(
                    token_pred.reshape(b * steps, width, dim),
                    predicted_mask.reshape(b * steps, width) & next_mask,
                    next_ids, sentences.shape[2],
                )
                base = base.reshape(b, steps, sentences.shape[2], dim)
            sentence_pred = self.sentence_pred(
                base.reshape(b * steps, sentences.shape[2], dim),
                sentence_mask[:, :-1].reshape(b * steps, sentences.shape[2]),
                action.reshape(b * steps, -1), affected.reshape(-1),
                sentences[:, :-1].reshape(b * steps, sentences.shape[2], dim)
                if self.variant not in {"sentence", "sentence_macro"} else None,
            ).reshape(b, steps, sentences.shape[2], dim)

        if sentence_pred is not None:
            global_pred = _masked_pool(sentence_pred, sentence_mask[:, :-1])
            global_states = _masked_pool(sentences, sentence_mask)
            global_targets = _masked_pool(tgt_sentences, tgt_sentence_mask)
        else:
            global_pred = _masked_pool(token_pred, predicted_mask)
            global_states = _masked_pool(tokens, token_mask)
            global_targets = _masked_pool(tgt_tokens, tgt_token_mask)
        step_mask = batch["step_mask"][:, :steps]
        rollout = global_pred  # one-step placeholder; explicit token rollout is separate
        value = self.value_head(global_states.detach()).squeeze(-1)
        zeros_ops = global_pred.new_zeros(b, steps, 3)
        maybe_detach = (
            (lambda value: value.detach())
            if self.detach_targets else (lambda value: value)
        )
        out = JEPAOutputs(
            s0=global_states[:, 0], step_states=global_states[:, 1:],
            prev_states=global_states[:, :-1],
            step_states_tgt=maybe_detach(global_targets[:, 1:]), actions=action,
            action_emb_tgt=global_pred.detach(), preds=global_pred,
            rollout=rollout, op_logits=zeros_ops,
            emb_pred=global_pred.new_zeros(global_pred.shape), value_pred=value,
            step_mask=step_mask,
        )
        out.extras.update({
            "multiscale_variant": self.variant,
            "token_predictions": token_pred if self.use_token_loss else None,
            "token_prediction_mask": predicted_mask,
            "token_targets": maybe_detach(tgt_tokens[:, 1:]),
            "token_target_mask": tgt_token_mask[:, 1:],
            "sentence_predictions": sentence_pred,
            "sentence_targets": maybe_detach(tgt_sentences[:, 1:]),
            "sentence_target_mask": tgt_sentence_mask[:, 1:],
            "affected_sentence": affected,
            "sentence_attention": attention,
            "token_states": tokens,
            "token_states_tgt": maybe_detach(tgt_tokens),
            "token_state_mask": token_mask,
            "sentence_states": sentences,
            "sentence_states_tgt": maybe_detach(tgt_sentences),
            "transition_slice_start": transition_start,
            "efficient_pair_encoding": use_pair_encoding,
            "target_encoder_mode": self.target_encoder_mode,
            "sigreg_states": global_states[:, :-1],
            "sigreg_state_mask": step_mask,
            "observed_action_targets": batch["action_tokens"][:, :steps],
        })
        if "goal_distance" in batch:
            out.extras["state_goal_distance_prediction"] = value
            raw_goal_distance = batch["goal_distance"][:, :states].float()
            out.extras["state_goal_distance_target"] = (
                raw_goal_distance
                / raw_goal_distance[:, :1].clamp_min(1)
            ).detach()
            out.extras["state_goal_distance_mask"] = torch.cat([
                torch.ones_like(step_mask[:, :1]), step_mask
            ], 1)
        pooled_current = global_states[:, :-1]
        out.extras["base_action_value"] = self.action_value(
            pooled_current, q_action
        )
        if "gar_token_edit_target" in batch:
            out.extras["base_action_value_target"] = (
                batch["gar_token_edit_target"][:, :steps].float().detach()
            )

        if self.base_prior is not None:
            flat_prompt = prompt_emb[:, None].expand(b, steps, dim).reshape(-1, dim)
            position_logits, content_logits = self.replacement_prior(
                current, current_mask, flat_prompt, pos.reshape(-1)
            )
            out.extras["refinement_content_logits"] = content_logits.reshape(
                b, steps, *content_logits.shape[1:]
            )
            if position_logits is not None:
                out.extras["refinement_position_logits"] = (
                    position_logits.reshape(b, steps, -1)
                )

        proposal_ops = batch.get("proposal_op")
        if proposal_ops is not None:
            proposal_steps = min(steps, proposal_ops.shape[1])
            candidates = proposal_ops.shape[2]
            base_states = tokens[:, :proposal_steps].reshape(
                b * proposal_steps, width, dim
            )
            base_masks = token_mask[:, :proposal_steps].reshape(
                b * proposal_steps, width
            )
            p_content = self.encoder.tok(
                batch["proposal_edit_content_token"][:, :proposal_steps]
            )
            p_ops = proposal_ops[:, :proposal_steps].reshape(
                b * proposal_steps, candidates
            )
            p_pos = batch["proposal_edit_position"][:, :proposal_steps].reshape(
                b * proposal_steps, candidates
            )
            p_content = p_content.reshape(
                b * proposal_steps, candidates, dim
            )
            if self.base_q_action is not None:
                p_actions = self.base_q_action.encode_candidates(
                    base_states, base_masks, p_ops, p_pos, p_content,
                )
            elif self.token_pred is None:
                if isinstance(self.sentence_action, SentencePatternActionEncoder):
                    # The blank-pattern encoder is candidate-specific but does
                    # not consume state values.  Only the compact masks and
                    # sentence labels are repeated.
                    p_masks = base_masks.unsqueeze(1).expand(
                        -1, candidates, -1
                    ).reshape(-1, width)
                    p_ids = ids[:, :proposal_steps].reshape(
                        b * proposal_steps, width
                    ).unsqueeze(1).expand(-1, candidates, -1).reshape(-1, width)
                    p_actions = self.sentence_action(
                        p_content.new_empty(0), p_masks, p_ids,
                        p_ops.reshape(-1), p_pos.reshape(-1),
                        p_content.reshape(-1, dim),
                    ).reshape(b * proposal_steps, candidates, -1)
                else:
                    p_actions = self.sentence_action.encode_candidates(
                        base_states, base_masks, p_ops, p_pos, p_content,
                    )
            else:
                p_actions = self.token_pred.encode_action_candidates(
                    base_states, base_masks, p_ops, p_pos, p_content,
                )
            p_actions = p_actions.reshape(b, proposal_steps, candidates, -1)
            p_global = global_states[:, :proposal_steps].unsqueeze(2).expand(
                -1, -1, candidates, -1
            )
            out.extras.update({
                "base_alt_action_value": self.action_value(p_global, p_actions),
                "base_alt_action_valid": (
                    batch["proposal_valid"][:, :proposal_steps]
                    & step_mask[:, :proposal_steps].unsqueeze(-1)
                ),
                "base_alt_action_target": batch[
                    "gar_proposal_token_edit_target"
                ][:, :proposal_steps].float().detach(),
            })
        if self.ldad is not None:
            row = torch.arange(b, device=tokens.device)[:, None]
            time = torch.arange(steps, device=tokens.device)[None, :]
            changed_next = sentences[row, time + 1, affected]
            changed_prev = sentences[row, time, affected]
            out.extras["observed_action_logits"] = self.ldad(
                changed_next - changed_prev
            )
            out.extras["ldad_uses_changed_sentence_delta"] = True

        if self.use_macro and steps >= self.macro_k:
            count = steps - self.macro_k + 1
            windows = torch.stack(
                [action[:, start:start + self.macro_k] for start in range(count)], 1
            )
            macro = self.macro_model(windows.reshape(-1, self.macro_k, action.shape[-1]))
            macro = macro.reshape(b, count, -1)
            start_states = sentences[:, :count]
            start_mask = sentence_mask[:, :count]
            macro_pred = self.macro_pred(
                start_states.reshape(-1, sentences.shape[2], dim),
                start_mask.reshape(-1, sentences.shape[2]),
                macro.reshape(-1, macro.shape[-1]), None,
            ).reshape(b, count, sentences.shape[2], dim)
            endpoint = tgt_sentences[:, self.macro_k:self.macro_k + count]
            endpoint_mask = tgt_sentence_mask[:, self.macro_k:self.macro_k + count]
            macro_valid = torch.stack([
                step_mask[:, start:start + self.macro_k].all(-1)
                for start in range(count)
            ], 1)
            out.hi_preds = _masked_pool(macro_pred, endpoint_mask)
            out.hi_targets = _masked_pool(endpoint, endpoint_mask).detach()
            out.hi_mask = macro_valid
            prior_state = _masked_pool(start_states, start_mask)
            if self.macro_prior_detach_state:
                # Prior fitting must learn a deployable proposal distribution,
                # not reshape the state encoder through a potentially negative
                # Gaussian log-density. Macro endpoint prediction remains the
                # representation-learning signal.
                prior_state = prior_state.detach()
            prior_mu, prior_logvar = self.macro_model.prior_params(prior_state)
            out.extras.update({
                "macro_codes": macro,
                "macro_sentence_predictions": macro_pred,
                "macro_sentence_targets": endpoint.detach(),
                "macro_sentence_mask": endpoint_mask & macro_valid[..., None],
                "macro_prior_mu": prior_mu,
                "macro_prior_logvar": prior_logvar,
                "macro_window_starts": torch.arange(count, device=tokens.device),
                "macro_window_endpoints": torch.arange(
                    self.macro_k, self.macro_k + count, device=tokens.device
                ),
            })
            if self.macro_decoder is not None:
                decoder_states = torch.stack([
                    torch.stack([
                        tokens[:, start + offset]
                        for offset in range(self.macro_k)
                    ], 1)
                    for start in range(count)
                ], 1)
                decoder_masks = torch.stack([
                    torch.stack([
                        token_mask[:, start + offset]
                        for offset in range(self.macro_k)
                    ], 1)
                    for start in range(count)
                ], 1)
                decoder_positions = torch.stack([
                    pos[:, start:start + self.macro_k]
                    for start in range(count)
                ], 1)
                decoder_macro = macro.unsqueeze(2).expand(
                    -1, -1, self.macro_k, -1
                )
                decoder_prompt = prompt_emb[:, None, None].expand(
                    -1, count, self.macro_k, -1
                )
                decoder_step = torch.arange(
                    self.macro_k, device=tokens.device
                ).view(1, 1, -1).expand(b, count, -1)
                decoder_position_logits, decoder_content_logits = (
                    self.macro_decoder(
                        decoder_states.reshape(-1, width, dim),
                        decoder_masks.reshape(-1, width),
                        decoder_prompt.reshape(-1, dim),
                        decoder_macro.reshape(-1, macro.shape[-1]),
                        decoder_step.reshape(-1),
                        decoder_positions.reshape(-1),
                    )
                )
                decoder_valid = macro_valid.unsqueeze(-1).expand(
                    -1, -1, self.macro_k
                )
                out.extras.update({
                    "macro_decoder_position_logits": (
                        decoder_position_logits.reshape(
                            b, count, self.macro_k, width
                        )
                    ),
                    "macro_decoder_content_logits": (
                        decoder_content_logits.reshape(
                            b, count, self.macro_k, -1
                        )
                    ),
                    "macro_decoder_position_target": decoder_positions,
                    "macro_decoder_content_target": torch.stack([
                        batch["edit_content_token"][
                            :, start:start + self.macro_k
                        ] for start in range(count)
                    ], 1),
                    "macro_decoder_valid": decoder_valid,
                })
        return out
