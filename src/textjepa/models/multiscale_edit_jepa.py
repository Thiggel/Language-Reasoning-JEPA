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
from textjepa.models.layers import encoder_stack
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
                 dropout: float = 0.0, pooling: str = "attention"):
        super().__init__()
        if pooling not in {"attention", "mean"}:
            raise ValueError(f"unknown sentence pooling: {pooling}")
        self.pooling = pooling
        self.pad_id = int(pad_id)
        self.max_sequence_len = int(max_sequence_len)
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.token_pos = nn.Parameter(torch.zeros(1, max_sequence_len, d_model))
        self.segment = nn.Parameter(torch.zeros(2, d_model))
        self.token_encoder = encoder_stack(
            d_model, token_layers, n_heads, ff_mult, dropout
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
            d_model, sentence_layers, n_heads, ff_mult, dropout
        )
        self.sentence_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.token_pos, std=0.02)
        nn.init.normal_(self.segment, std=0.02)
        nn.init.normal_(self.sentence_pos, std=0.02)

    @staticmethod
    def _pack(prompt: torch.Tensor, buffer: torch.Tensor, pad_id: int):
        """Pack valid tokens and retain -1(prompt)/sentence buffer labels."""
        n, c, length = buffer.shape
        rows, labels = [], []
        widths = []
        for row in range(n):
            p = prompt[row].reshape(-1)
            p = p[p.ne(pad_id)]
            pieces, ids = [p], [torch.full_like(p, -1)]
            for sentence in range(c):
                value = buffer[row, sentence]
                value = value[value.ne(pad_id)]
                pieces.append(value)
                ids.append(torch.full_like(value, sentence))
            packed = torch.cat(pieces) if pieces else prompt.new_empty(0)
            label = torch.cat(ids) if ids else prompt.new_empty(0)
            rows.append(packed)
            labels.append(label)
            widths.append(max(int(packed.numel()), 1))
        width = max(widths)
        tokens = prompt.new_full((n, width), pad_id)
        sentence_ids = prompt.new_full((n, width), -2)
        valid = torch.zeros(n, width, dtype=torch.bool, device=prompt.device)
        for row, (values, ids) in enumerate(zip(rows, labels)):
            tokens[row, :values.numel()] = values
            sentence_ids[row, :ids.numel()] = ids
            valid[row, :values.numel()] = True
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
        key_pad = ~valid
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        h = self.token_norm(self.token_encoder(h, src_key_padding_mask=key_pad))
        buffer_valid = valid & sentence_ids.ge(0)
        widths = buffer_valid.sum(-1).clamp_min(1)
        width = int(widths.max().item())
        out = h.new_zeros(h.shape[0], width, h.shape[-1])
        out_ids = sentence_ids.new_full((h.shape[0], width), -1)
        out_mask = torch.zeros(
            h.shape[0], width, dtype=torch.bool, device=h.device
        )
        for row in range(h.shape[0]):
            keep = buffer_valid[row]
            count = int(keep.sum().item())
            out[row, :count] = h[row, keep]
            out_ids[row, :count] = sentence_ids[row, keep]
            out_mask[row, :count] = True
        return out, out_mask, out_ids

    def pool_sentences(self, token_states: torch.Tensor,
                       token_mask: torch.Tensor, sentence_ids: torch.Tensor,
                       n_sentences: int):
        if n_sentences > self.sentence_pos.shape[1]:
            raise ValueError("too many sentences for configured sentence positions")
        n, width, dim = token_states.shape
        sentence_mask = torch.zeros(
            n, n_sentences, dtype=torch.bool, device=token_states.device
        )
        pooled = token_states.new_zeros(n, n_sentences, dim)
        attention = token_states.new_zeros(n, width)
        raw_score = self.pool_score(token_states).squeeze(-1)
        for sentence in range(n_sentences):
            members = token_mask & sentence_ids.eq(sentence)
            sentence_mask[:, sentence] = members.any(-1)
            if self.pooling == "attention":
                score = raw_score.masked_fill(~members, -torch.inf)
                # Avoid NaNs for absent/padded sentences; output is masked.
                score = torch.where(members.any(-1, keepdim=True), score,
                                    torch.zeros_like(score))
                weight = torch.softmax(score, -1) * members.to(score.dtype)
            else:
                weight = members.to(raw_score.dtype)
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1)
            attention = attention + weight
            pooled[:, sentence] = torch.einsum("nw,nwd->nd", weight, token_states)
        key_pad = ~sentence_mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        encoded = self.sentence_encoder(
            pooled + self.sentence_pos[:, :n_sentences],
            src_key_padding_mask=key_pad,
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
                 n_heads: int = 8, correction: bool = False):
        super().__init__()
        self.correction = correction
        self.action = nn.Linear(d_action, d_model)
        self.current = nn.Linear(d_model, d_model) if correction else None
        self.blocks = encoder_stack(d_model, n_layers, n_heads, 4, 0.0)
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
            local[row, index] = cond
            h = h + local
        key_pad = ~mask
        key_pad = key_pad.clone()
        key_pad[key_pad.all(-1), 0] = False
        delta = self.out(self.blocks(h, src_key_padding_mask=key_pad))
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


class TokenReplacementPrior(nn.Module):
    """Factorized deployment prior over ``position`` then ``token``.

    The prior consumes only the current online state and prompt.  In
    particular, neither the clean buffer nor an EMA goal is an input.  The
    optional stop-gradient is the clean ablation between a read-only policy
    head and a policy loss that is allowed to shape the representation.
    """

    def __init__(self, d_model: int, vocab_size: int, detach_state: bool):
        super().__init__()
        self.detach_state = bool(detach_state)
        self.position = nn.Sequential(
            nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model),
            nn.GELU(), nn.Linear(d_model, 1),
        )
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
        position_logits = self.position(shared).squeeze(-1)
        position_logits = position_logits.masked_fill(~mask, -torch.inf)
        row = torch.arange(len(states), device=states.device)
        selected = states[
            row, positions.clamp(0, states.shape[1] - 1)
        ]
        content_logits = self.content(torch.cat([selected, pooled, prompt], -1))
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
                 macro_decoder_detach_inputs: bool = True):
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown multiscale edit variant: {variant}")
        if dropout != 0:
            raise ValueError("multiscale edit JEPA requires dropout=0")
        self.variant = variant
        self.use_token_loss = variant not in {"sentence", "sentence_macro"}
        self.use_sentence = variant != "token"
        self.use_macro = variant in {"sentence_macro", "token_sentence_macro"}
        self.macro_k = int(macro_k)
        self.macro_prior_detach_state = bool(macro_prior_detach_state)
        self.max_transitions_per_forward = max(
            0, int(max_transitions_per_forward)
        )
        self.encoder = HierarchicalBufferEncoder(
            vocab_size, pad_id, d_model, token_layers, sentence_layers,
            n_heads, ff_mult, max_sequence_len, max_sentences, dropout,
            sentence_pooling,
        )
        self.teacher = EMATeacher(self.encoder)
        self.token_pred = None if variant in {"sentence", "sentence_macro"} else TokenAlignedEditPredictor(
            d_model, d_action, predictor_layers, n_heads,
            relative_radius=token_relative_radius,
        )
        self.sentence_action = PrimitiveEditActionEncoder(
            d_model, d_action
        ) if variant in {"sentence", "sentence_macro"} else None
        self.sentence_pred = None if not self.use_sentence else SentenceEditPredictor(
            d_model, d_action, predictor_layers, n_heads,
            correction=variant not in {"sentence", "sentence_macro"},
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
                d_model, d_macro, predictor_layers, n_heads
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
            d_model, vocab_size, base_prior_detach_state
        ) if base_prior else None
        self.base_q_head = nn.Sequential(
            nn.LayerNorm(d_model + d_action),
            nn.Linear(d_model + d_action, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
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
        self.teacher.update(self.encoder, momentum)

    @staticmethod
    def affected_sentences(ids: torch.Tensor, mask: torch.Tensor,
                           operations: torch.Tensor, positions: torch.Tensor):
        """Map a pointer/gap to a sentence without a mutable absolute register."""
        result = torch.zeros_like(positions)
        for row in range(len(ids)):
            length = int(mask[row].sum().item())
            if length == 0:
                continue
            pos = int(positions[row].item())
            op = int(operations[row].item())
            if op == 1:  # insert: a gap belongs to the sentence on its right
                pos = min(max(pos, 0), length - 1)
            else:
                pos = min(max(pos, 0), length - 1)
            result[row] = ids[row, pos].clamp_min(0)
        return result

    @staticmethod
    def transition_sentence_ids(ids: torch.Tensor, mask: torch.Tensor,
                                operations: torch.Tensor,
                                positions: torch.Tensor):
        """Apply the same structural edit as the token predictor to labels."""
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

    def _encode_trajectory(self, batch: dict, teacher: bool = False):
        b, states, sentences, length = batch["buffer_tokens"].shape
        prompt = batch["prompt_tokens"].unsqueeze(1).expand(
            b, states, *batch["prompt_tokens"].shape[1:]
        )
        module = self.teacher if teacher else self.encoder
        result = module(
            prompt.reshape(b * states, *prompt.shape[2:]),
            batch["buffer_tokens"].reshape(b * states, sentences, length),
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
        tokens, token_mask, ids, sentences, sentence_mask, attention = (
            self._encode_trajectory(batch)
        )
        with torch.no_grad():
            tgt_tokens, tgt_token_mask, _, tgt_sentences, tgt_sentence_mask, _ = (
                self._encode_trajectory(batch, teacher=True)
            )
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
        action_module = self.sentence_action if self.token_pred is None else self.token_pred
        action_fn = action_module if self.token_pred is None else action_module.encode_action
        action = action_fn(
            current, current_mask, op.reshape(-1), pos.reshape(-1),
            content.reshape(-1, dim),
        ).reshape(b, steps, -1)
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
        out = JEPAOutputs(
            s0=global_states[:, 0], step_states=global_states[:, 1:],
            prev_states=global_states[:, :-1],
            step_states_tgt=global_targets[:, 1:].detach(), actions=action,
            action_emb_tgt=global_pred.detach(), preds=global_pred,
            rollout=rollout, op_logits=zeros_ops,
            emb_pred=global_pred.new_zeros(global_pred.shape), value_pred=value,
            step_mask=step_mask,
        )
        out.extras.update({
            "multiscale_variant": self.variant,
            "token_predictions": token_pred if self.use_token_loss else None,
            "token_prediction_mask": predicted_mask,
            "token_targets": tgt_tokens[:, 1:].detach(),
            "token_target_mask": tgt_token_mask[:, 1:],
            "sentence_predictions": sentence_pred,
            "sentence_targets": tgt_sentences[:, 1:].detach(),
            "sentence_target_mask": tgt_sentence_mask[:, 1:],
            "affected_sentence": affected,
            "sentence_attention": attention,
            "token_states": tokens,
            "token_states_tgt": tgt_tokens.detach(),
            "token_state_mask": token_mask,
            "sentence_states": sentences,
            "sentence_states_tgt": tgt_sentences.detach(),
            "transition_slice_start": transition_start,
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
            pooled_current, action
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
            out.extras.update({
                "refinement_position_logits": position_logits.reshape(
                    b, steps, -1
                ),
                "refinement_content_logits": content_logits.reshape(
                    b, steps, -1
                ),
            })

        proposal_ops = batch.get("proposal_op")
        if proposal_ops is not None:
            proposal_steps = min(steps, proposal_ops.shape[1])
            candidates = proposal_ops.shape[2]
            p_states = tokens[:, :proposal_steps].unsqueeze(2).expand(
                -1, -1, candidates, -1, -1
            )
            p_masks = token_mask[:, :proposal_steps].unsqueeze(2).expand(
                -1, -1, candidates, -1
            )
            p_content = self.encoder.tok(
                batch["proposal_edit_content_token"][:, :proposal_steps]
            )
            p_action_fn = (
                self.sentence_action if self.token_pred is None
                else self.token_pred.encode_action
            )
            p_actions = p_action_fn(
                p_states.reshape(-1, width, dim),
                p_masks.reshape(-1, width),
                proposal_ops[:, :proposal_steps].reshape(-1),
                batch["proposal_edit_position"][:, :proposal_steps].reshape(-1),
                p_content.reshape(-1, dim),
            ).reshape(b, proposal_steps, candidates, -1)
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
