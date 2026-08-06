"""Token-action JEPA with an evolving attention-pooled sentence state."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from textjepa.models.ema import EMATeacher
from textjepa.models.heads import MacroValueHead
from textjepa.models.layers import mlp
from textjepa.models.predictor import CausalHistoryPredictor
from textjepa.models.token_hierarchy import CausalTokenStateEncoder
from textjepa.objectives.visreg import VISReg


class CausalAttentionPooler(nn.Module):
    """Pool every causal prefix, optionally resetting after punctuation."""

    def __init__(
        self, d_state: int, n_heads: int, scope: str,
        boundary_ids: tuple[int, ...],
    ):
        super().__init__()
        if scope not in {"sentence", "global"}:
            raise ValueError("pooling scope must be sentence or global")
        if d_state % n_heads:
            raise ValueError("state width must be divisible by pool heads")
        self.scope = scope
        self.boundary_ids = tuple(map(int, boundary_ids))
        self.n_heads = int(n_heads)
        self.head_dim = d_state // n_heads
        self.query = nn.Linear(d_state, d_state)
        self.key = nn.Linear(d_state, d_state)
        self.value = nn.Linear(d_state, d_state)
        self.output = nn.Linear(d_state, d_state)
        self.norm = nn.LayerNorm(d_state)
        self.query_bias = nn.Parameter(torch.zeros(1, 1, d_state))

    def _allowed(self, tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        batch, length = tokens.shape
        valid = tokens.ne(pad_id)
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril()
        allowed = causal[None] & valid[:, None, :]
        if self.scope == "sentence":
            boundary = torch.zeros_like(valid)
            for token_id in self.boundary_ids:
                boundary |= tokens.eq(token_id)
            # A punctuation token closes its current segment; the following
            # token begins the next segment.
            segment = boundary.long().cumsum(1) - boundary.long()
            allowed &= segment[:, :, None].eq(segment[:, None, :])
        # Avoid all-masked padded query rows; outputs are zeroed afterwards.
        safe_index = torch.arange(length, device=tokens.device)
        allowed[:, safe_index, safe_index] |= ~valid
        return allowed

    def forward(self, hidden: torch.Tensor, tokens: torch.Tensor, pad_id: int):
        batch, length, dim = hidden.shape
        def heads(value):
            return value.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        query = heads(self.query(hidden + self.query_bias))
        key, value = heads(self.key(hidden)), heads(self.value(hidden))
        allowed = self._allowed(tokens, pad_id)[:, None]
        # Let PyTorch dispatch to its fused scaled-dot-product-attention
        # kernels.  The previous explicit QK^T/softmax implementation kept a
        # full [B,H,L,L] score tensor alive and dominated memory at scale.
        pooled = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0,
        )
        pooled = pooled.transpose(1, 2).reshape(batch, length, dim)
        pooled = self.norm(self.output(pooled))
        return pooled.masked_fill(tokens.eq(pad_id).unsqueeze(-1), 0.0)


class PooledCausalStateEncoder(nn.Module):
    def __init__(
        self, vocab_size, pad_id, period_id, question_id, d_state,
        encoder_layers, n_heads, ff_mult, max_len, pool_heads, pooling_scope,
        attention_backend="auto", sequence_packing=False,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.backbone = CausalTokenStateEncoder(
            vocab_size, pad_id, d_state, encoder_layers, n_heads, ff_mult,
            max_len, attention_backend, sequence_packing,
        )
        self.pooler = CausalAttentionPooler(
            d_state, pool_heads, pooling_scope, (period_id, question_id)
        )

    def forward(self, tokens):
        return self.pooler(self.backbone(tokens), tokens, self.pad_id)


class PrefixAutoregressiveDecoder(nn.Module):
    """Teacher-forced causal decoder cross-attending to one pooled state."""

    def __init__(
        self, vocab_size, pad_id, d_state, d_model=256, n_layers=2,
        n_heads=8, ff_mult=4, max_len=96,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.max_len = int(max_len)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        self.memory = nn.Sequential(nn.LayerNorm(d_state), nn.Linear(d_state, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_model * ff_mult, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerDecoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, state, target_ids, valid):
        count, length = target_ids.shape
        shifted = self.embedding(target_ids[:, :-1]) if length > 1 else self.bos[:, :0].expand(count, 0, -1)
        inp = torch.cat([self.bos.expand(count, -1, -1), shifted], 1)
        inp = inp + self.pos[:, :length]
        causal = torch.ones(length, length, dtype=torch.bool, device=state.device).triu(1)
        safe_valid = valid.clone()
        safe_valid[:, 0] = True
        decoded = self.blocks(
            inp, self.memory(state)[:, None], tgt_mask=causal,
            tgt_key_padding_mask=~safe_valid,
        )
        return self.output(self.norm(decoded))


class PooledSentenceJEPA(nn.Module):
    def __init__(
        self, vocab_size: int, pad_id: int, period_id: int, question_id: int,
        d_state: int = 512, encoder_layers: int = 8, pool_heads: int = 8,
        predictor_layers: int = 4, n_heads: int = 8, ff_mult: int = 4,
        max_len: int = 768, d_action: int = 128, dense_depth: int = 4,
        dense_checkpoint: bool = False,
        pooling_scope: str = "sentence", use_token_prior: bool = True,
        token_prior_hidden: int = 0, token_prior_detach_state: bool = False,
        use_prefix_decoder: bool = False, decoder_dim: int = 256,
        decoder_layers: int = 2, decoder_heads: int = 8,
        decoder_max_len: int = 96, decoder_prefixes_per_sequence: int = 8,
        target_mode: str = "ema", visreg_projections: int = 4096,
        attention_backend: str = "auto", sequence_packing: bool = False,
    ):
        super().__init__()
        if target_mode not in {"ema", "visreg"}:
            raise ValueError("target_mode must be 'ema' or 'visreg'")
        self.target_mode = target_mode
        self.pad_id, self.period_id = int(pad_id), int(period_id)
        self.question_id = int(question_id)
        self.vocab_size, self.d_state, self.d_action = int(vocab_size), int(d_state), int(d_action)
        self.dense_depth = max(1, int(dense_depth))
        self.dense_checkpoint = bool(dense_checkpoint)
        self.token_prior_detach_state = bool(token_prior_detach_state)
        self.decoder_prefixes_per_sequence = int(decoder_prefixes_per_sequence)
        self.state_encoder = PooledCausalStateEncoder(
            vocab_size, pad_id, period_id, question_id, d_state,
            encoder_layers, n_heads, ff_mult, max_len, pool_heads,
            pooling_scope, attention_backend, sequence_packing,
        )
        self.teacher = (
            EMATeacher(self.state_encoder) if target_mode == "ema" else None
        )
        self.visreg = VISReg(visreg_projections) if target_mode == "visreg" else None
        self.token_action = nn.Embedding(vocab_size, d_action, padding_idx=pad_id)
        self.predictor = CausalHistoryPredictor(
            d_state, d_action, predictor_layers, n_heads, ff_mult,
            max_steps=max_len, residual=True,
            attention_backend=attention_backend,
            sequence_packing=sequence_packing,
        )
        self.goal_head = mlp([d_state, 2 * d_state], d_state)
        self.token_value = MacroValueHead(d_state, d_action)
        if use_token_prior:
            layers: list[nn.Module] = [nn.LayerNorm(d_state)]
            if token_prior_hidden:
                layers += [
                    nn.Linear(d_state, token_prior_hidden), nn.GELU(),
                    nn.Linear(token_prior_hidden, vocab_size),
                ]
            else:
                layers.append(nn.Linear(d_state, vocab_size))
            self.token_prior = nn.Sequential(*layers)
        else:
            self.token_prior = None
        self.prefix_decoder = PrefixAutoregressiveDecoder(
            vocab_size, pad_id, d_state, decoder_dim, decoder_layers,
            decoder_heads, ff_mult, decoder_max_len,
        ) if use_prefix_decoder else None

    @torch.no_grad()
    def update_teacher(self, momentum):
        if self.teacher is not None:
            self.teacher.update(self.state_encoder, momentum)

    def _sequences(self, states, targets, tokens, prompt_len):
        batch, _, dim = states.shape
        lengths = (tokens.ne(self.pad_id).sum(1) - prompt_len).clamp_min(1)
        width = int(lengths.max())
        offset = torch.arange(width, device=tokens.device)[None]
        valid = offset < lengths[:, None]
        target_index = prompt_len[:, None] + offset
        safe_target_index = target_index.clamp_max(tokens.shape[1] - 1)
        state_index = safe_target_index[..., None].expand(-1, -1, dim)
        prev_index = (target_index - 1).clamp_max(tokens.shape[1] - 1)
        prev_index = prev_index[..., None].expand(-1, -1, dim)
        prev = states.gather(1, prev_index).masked_fill(~valid[..., None], 0.0)
        target = targets.gather(1, state_index).masked_fill(~valid[..., None], 0.0)
        ids = tokens.gather(1, safe_target_index).masked_fill(~valid, self.pad_id)
        row = torch.arange(batch, device=tokens.device)
        prompt_state = states[row, prompt_len - 1]
        final_target = targets[row, prompt_len + lengths - 1]
        return {
            "prev": prev, "target": target, "action_ids": ids,
            "valid": valid, "lengths": lengths,
            "prompt_state": prompt_state,
            "final_target": final_target,
        }

    @staticmethod
    def _dense_all_starts(
        predictor, first, targets, prev, actions, valid, depth,
        checkpoint_predictor=False,
    ):
        """Open-loop endpoints from every sequence position.

        For anchor ``t`` and horizon ``h``, retain the observed causal history
        through ``s_t`` and recursively insert predictions for actions
        ``a_t ... a_{t+h-1}``.  This differs from shifting a matrix of
        independent one-step predictions, which incorrectly treats predictions
        made from other anchors as one coherent trajectory.
        """
        batch, width, dim = prev.shape
        limit = min(max(1, int(depth)), width)
        predictions = [first]
        shifted_targets = [targets]
        masks = [valid]
        # paths[b, t] stores the recursively predicted states after anchor t.
        paths = first.unsqueeze(2)
        for horizon in range(2, limit + 1):
            anchors = width - horizon + 1
            starts = torch.arange(anchors, device=prev.device)
            lengths = starts + horizon
            # [B,A,W,D] is the same padded batch previously assembled by
            # B*A Python loops, but constructed with vectorized views/writes.
            padded_states = prev[:, None].expand(-1, anchors, -1, -1).clone()
            for offset in range(horizon - 1):
                padded_states[:, starts, starts + 1 + offset] = paths[:, :anchors, offset]
            padded_states = padded_states.reshape(batch * anchors, width, dim)
            padded_actions = actions[:, None].expand(
                -1, anchors, -1, -1
            ).reshape(batch * anchors, width, actions.shape[-1])
            sequence_valid = torch.arange(width, device=prev.device)[None] < (
                lengths.repeat(batch)[:, None]
            )
            if checkpoint_predictor and torch.is_grad_enabled():
                all_predictions = checkpoint(
                    predictor, padded_states, padded_actions, sequence_valid,
                    use_reentrant=False,
                )
            else:
                all_predictions = predictor(
                    padded_states, padded_actions, sequence_valid
                )
            gather = lengths.repeat(batch) - 1
            endpoint = all_predictions[
                torch.arange(batch * anchors, device=prev.device), gather
            ].reshape(batch, anchors, dim)
            anchor_mask = valid.unfold(1, horizon, 1).all(-1)
            predictions.append(endpoint)
            shifted_targets.append(targets[:, horizon - 1:])
            masks.append(anchor_mask)
            paths = torch.cat([
                paths[:, :anchors], endpoint.unsqueeze(2)
            ], 2)
        return tuple(predictions), tuple(shifted_targets), tuple(masks)

    def forward(self, tokens, prompt_len, sentence_ends=None):
        states = self.state_encoder(tokens)
        if self.teacher is None:
            # Heuristic-free VISReg: the future target is another position from
            # the same online encoder and receives prediction-loss gradients.
            targets = states
        else:
            with torch.no_grad():
                targets = self.teacher(tokens)
        seq = self._sequences(states, targets, tokens, prompt_len)
        actions = self.token_action(seq["action_ids"])
        pred = self.predictor(seq["prev"], actions, seq["valid"])
        dense = self._dense_all_starts(
            self.predictor, pred, seq["target"], seq["prev"], actions,
            seq["valid"], self.dense_depth, self.dense_checkpoint,
        )
        prior_state = seq["prev"].detach() if self.token_prior_detach_state else seq["prev"]
        logits = self.token_prior(prior_state) if self.token_prior else None
        prompt = seq["prompt_state"][:, None].expand_as(seq["prev"])
        return {
            **seq, "states": states, "targets": targets, "actions": actions,
            "pred": pred, "dense_predictions": dense[0],
            "dense_targets": dense[1], "dense_masks": dense[2],
            "token_prior_logits": logits,
            "value": self.token_value(seq["prev"], prompt, actions),
            "goal_pred": self.goal_head(seq["prompt_state"]),
        }

    def prefix_decoder_batch(self, out, tokens, prompt_len, sentence_ends=None):
        if self.prefix_decoder is None:
            raise ValueError("prefix decoder is disabled")
        states, prefixes = [], []
        for row in range(tokens.shape[0]):
            count = int(out["lengths"][row])
            number = min(self.decoder_prefixes_per_sequence, count)
            endpoints = torch.linspace(0, count - 1, number, device=tokens.device).long().unique()
            prompt = int(prompt_len[row])
            for endpoint_t in endpoints:
                endpoint = prompt + int(endpoint_t)
                earlier = tokens[row, :endpoint]
                boundary = earlier.eq(self.period_id) | earlier.eq(self.question_id)
                indices = boundary.nonzero(as_tuple=False)
                start = int(indices[-1]) + 1 if len(indices) else 0
                prefix = tokens[row, start:endpoint + 1]
                prefix = prefix[-self.prefix_decoder.max_len:]
                states.append(out["states"][row, endpoint])
                prefixes.append(prefix)
        width = max(map(len, prefixes))
        ids = tokens.new_full((len(prefixes), width), self.pad_id)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for index, prefix in enumerate(prefixes):
            ids[index, :len(prefix)] = prefix
            valid[index, :len(prefix)] = True
        memory = torch.stack(states)
        logits = self.prefix_decoder(memory, ids, valid)
        permutation = torch.roll(torch.arange(len(memory), device=tokens.device), 1)
        shuffled = self.prefix_decoder(memory[permutation], ids, valid)
        return {
            "logits": logits, "shuffled_logits": shuffled,
            "targets": ids, "valid": valid, "states": memory,
        }

    def token_counterfactuals(self, out, tokens, prompt_len, k=8, max_anchors=16):
        anchors = out["valid"].nonzero(as_tuple=False)
        if len(anchors) > int(max_anchors):
            pick = torch.linspace(0, len(anchors) - 1, int(max_anchors), device=tokens.device).long().unique()
            anchors = anchors[pick]
        n = len(anchors)
        k = max(1, min(int(k), self.vocab_size - 1))
        factual = out["action_ids"][anchors[:, 0], anchors[:, 1]]
        offsets = torch.linspace(0, self.vocab_size - 1, k, device=tokens.device).long()[None]
        candidate_ids = (factual[:, None] + offsets) % self.vocab_size
        candidate_ids[:, 0] = factual
        candidate_ids[candidate_ids.eq(self.pad_id)] = 1
        candidate_ids[:, 0] = factual
        candidate_actions = self.token_action(candidate_ids)
        sequences, lengths = [], []
        for anchor_index, (row_t, column_t) in enumerate(anchors):
            row, column = int(row_t), int(column_t)
            source = int(prompt_len[row]) - 1 + column
            prefix = tokens[row, :source + 1]
            for candidate in candidate_ids[anchor_index]:
                sequence = torch.cat([prefix, candidate.view(1)])
                sequences.append(sequence); lengths.append(len(sequence))
        padded = tokens.new_full((n * k, max(lengths)), self.pad_id)
        for index, sequence in enumerate(sequences):
            padded[index, :len(sequence)] = sequence
        with torch.no_grad():
            target_encoder = self.teacher if self.teacher is not None else self.state_encoder
            encoded = target_encoder(padded)
            exact = torch.stack([
                encoded[index, length - 1] for index, length in enumerate(lengths)
            ]).reshape(n, k, self.d_state)
        max_history = int(anchors[:, 1].max()) + 1
        hs = out["prev"].new_zeros(n * k, max_history, self.d_state)
        ha = candidate_actions.new_zeros(n * k, max_history, self.d_action)
        hv = torch.zeros(n * k, max_history, dtype=torch.bool, device=tokens.device)
        gather = []
        for anchor_index, (row_t, column_t) in enumerate(anchors):
            row, column = int(row_t), int(column_t); length = column + 1
            for candidate in range(k):
                flat = anchor_index * k + candidate
                hs[flat, :length] = out["prev"][row, :length]
                if column: ha[flat, :column] = out["actions"][row, :column]
                ha[flat, column] = candidate_actions[anchor_index, candidate]
                hv[flat, :length] = True; gather.append(column)
        prediction = self.predictor(hs, ha, hv)
        gather = torch.tensor(gather, device=tokens.device)
        prediction = prediction[torch.arange(n * k, device=tokens.device), gather].reshape(n, k, self.d_state)
        rows, columns = anchors[:, 0], anchors[:, 1]
        state, prompt, goal = out["prev"][rows, columns], out["prompt_state"][rows], out["final_target"][rows]
        value = self.token_value(
            state[:, None].expand(-1, k, -1), prompt[:, None].expand(-1, k, -1), candidate_actions
        )
        norm = lambda x: torch.nn.functional.layer_norm(x, (self.d_state,))
        before = (norm(state) - norm(goal)).square().mean(-1)
        after = (norm(exact) - norm(goal[:, None])).square().mean(-1)
        return {
            "anchor_indices": anchors, "candidate_ids": candidate_ids,
            "exact_outcome": exact, "predicted_outcome": prediction,
            "advantage_target": before[:, None] - after, "value": value,
            "candidate_valid": torch.ones_like(value, dtype=torch.bool),
        }
