"""Two-level token/sentence JEPA with distinct causal state spaces."""

from __future__ import annotations

import math

import torch
from torch import nn

from textjepa.models.action import MacroActionModel
from textjepa.models.ema import EMATeacher
from textjepa.models.heads import MacroSupportHead, MacroValueHead, ValueHead
from textjepa.models.layers import mlp
from textjepa.models.predictor import CausalHistoryPredictor
from textjepa.models.token_hierarchy import CausalTokenStateEncoder
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


class CrossLevelReachabilityHead(nn.Module):
    """Predict whether a high-level subgoal is executable from a low state."""

    def __init__(self, d_low: int, d_high: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_low + d_high),
            mlp([d_low + d_high, 2 * max(d_low, d_high)], 1),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([low, high], -1)).squeeze(-1)


class SentenceHierarchyJEPA(nn.Module):
    """A causal token model plus one semantic sentence hierarchy.

    The token and sentence encoders have independent parameters and output
    different-dimensional spaces. Sentence states are selected at the last
    token of each completed sentence; under causal attention this is exactly
    the boundary representation after observing the whole sentence. The
    observed macro action is a bidirectional CLS encoding of every token
    action inside that sentence.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_low: int = 256,
        d_high: int = 192,
        encoder_layers: int = 4,
        high_encoder_layers: int = 4,
        predictor_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_len: int = 768,
        d_token_action: int = 64,
        d_macro: int = 32,
        macro_layers: int = 2,
        macro_hidden: int = 128,
        macro_heads: int = 4,
        low_dense_depth: int = 2,
        high_dense_depth: int = 2,
        separate_high_encoder: bool = True,
        use_token_prior: bool = True,
        token_prior_hidden: int = 0,
        token_prior_detach_state: bool = False,
    ):
        super().__init__()
        if d_low % n_heads or d_high % n_heads:
            raise ValueError("both state widths must be divisible by n_heads")
        if macro_hidden % macro_heads:
            raise ValueError("macro_hidden must be divisible by macro_heads")
        self.pad_id = int(pad_id)
        self.d_low = int(d_low)
        self.d_high = int(d_high)
        self.d_macro = int(d_macro)
        self.low_dense_depth = max(1, int(low_dense_depth))
        self.high_dense_depth = max(1, int(high_dense_depth))
        self.token_prior_detach_state = bool(token_prior_detach_state)

        self.encoder = CausalTokenStateEncoder(
            vocab_size, pad_id, d_low, encoder_layers, n_heads, ff_mult, max_len
        )
        self.teacher = EMATeacher(self.encoder)
        self.separate_high_encoder = bool(separate_high_encoder)
        if self.separate_high_encoder:
            self.high_encoder = CausalTokenStateEncoder(
                vocab_size, pad_id, d_high, high_encoder_layers, n_heads,
                ff_mult, max_len,
            )
            self.high_teacher = EMATeacher(self.high_encoder)
        else:
            if d_high != d_low or high_encoder_layers != encoder_layers:
                raise ValueError(
                    "shared-state control requires matching state widths and encoder depths"
                )
            self.high_encoder = self.encoder
            self.high_teacher = self.teacher
        self.token_action = nn.Embedding(
            vocab_size, d_token_action, padding_idx=pad_id
        )
        self.low_predictor = CausalHistoryPredictor(
            d_low, d_token_action, predictor_layers, n_heads, ff_mult,
            max_steps=max_len, residual=False,
        )
        self.macro_action = MacroActionModel(
            d_token_action, d_high, d_macro, span=64, kind="transformer",
            variational=False, transformer_hidden=macro_hidden,
            transformer_layers=macro_layers, transformer_heads=macro_heads,
            transformer_max_actions=max_len,
        )
        self.high_predictor = CausalHistoryPredictor(
            d_high, d_macro, predictor_layers, n_heads, ff_mult,
            max_steps=max_len, residual=False,
        )
        self.low_to_high = nn.Sequential(nn.LayerNorm(d_low), nn.Linear(d_low, d_high))
        self.high_to_low = nn.Sequential(nn.LayerNorm(d_high), nn.Linear(d_high, d_low))
        self.planning_projection = nn.Sequential(
            nn.LayerNorm(d_high), nn.Linear(d_high, min(64, d_high))
        )
        self.low_value = ValueHead(d_low)
        self.high_value = ValueHead(d_high)
        self.macro_value = MacroValueHead(d_high, d_macro)
        self.macro_support = MacroSupportHead(d_high, d_macro)
        self.reachability = CrossLevelReachabilityHead(d_low, d_high)
        self.low_goal_head = mlp([d_low, 2 * d_low], d_low)
        self.high_goal_head = mlp([d_high, 2 * d_high], d_high)
        if use_token_prior:
            layers: list[nn.Module] = [nn.LayerNorm(d_low)]
            if token_prior_hidden:
                layers.extend([
                    nn.Linear(d_low, token_prior_hidden), nn.GELU(),
                    nn.Linear(token_prior_hidden, vocab_size),
                ])
            else:
                layers.append(nn.Linear(d_low, vocab_size))
            self.token_prior = nn.Sequential(*layers)
        else:
            self.token_prior = None

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        self.teacher.update(self.encoder, momentum)
        if self.separate_high_encoder:
            self.high_teacher.update(self.high_encoder, momentum)

    def _reasoning_sequences(self, states, targets, tokens, prompt_len):
        batch, _, dim = states.shape
        lengths = (tokens.ne(self.pad_id).sum(1) - prompt_len).clamp_min(1)
        width = int(lengths.max())
        prev = states.new_zeros(batch, width, dim)
        target = targets.new_zeros(batch, width, dim)
        ids = tokens.new_full((batch, width), self.pad_id)
        valid = torch.zeros(batch, width, dtype=torch.bool, device=tokens.device)
        prompt_state, final_target = [], []
        for row in range(batch):
            start, count = int(prompt_len[row]), int(lengths[row])
            prev[row, :count] = states[row, start - 1:start - 1 + count]
            target[row, :count] = targets[row, start:start + count]
            ids[row, :count] = tokens[row, start:start + count]
            valid[row, :count] = True
            prompt_state.append(states[row, start - 1])
            final_target.append(targets[row, start + count - 1])
        return {
            "prev": prev, "target": target, "action_ids": ids,
            "valid": valid, "lengths": lengths,
            "prompt_state": torch.stack(prompt_state),
            "final_target": torch.stack(final_target),
        }

    def _sentence_sequences(
        self, high_states, high_targets, seq, token_actions, prompt_len,
        sentence_ends,
    ):
        batch = high_states.shape[0]
        groups: list[list[tuple[int, int]]] = []
        for row in range(batch):
            ends = [
                int(end) for end in sentence_ends[row]
                if 0 < int(end) <= int(seq["lengths"][row])
            ]
            groups.append(list(zip([0] + ends[:-1], ends)))
        width = max(1, max(map(len, groups), default=0))
        raw_width = max(
            1, max((end - start for row in groups for start, end in row), default=0)
        )
        prev = high_states.new_zeros(batch, width, self.d_high)
        target = high_targets.new_zeros(batch, width, self.d_high)
        low_start = seq["prev"].new_zeros(batch, width, self.d_low)
        low_target = seq["target"].new_zeros(batch, width, self.d_low)
        raw_actions = token_actions.new_zeros(
            batch, width, raw_width, token_actions.shape[-1]
        )
        raw_ids = seq["action_ids"].new_full(
            (batch, width, raw_width), self.pad_id
        )
        raw_valid = torch.zeros(
            batch, width, raw_width, dtype=torch.bool, device=high_states.device
        )
        valid = torch.zeros(batch, width, dtype=torch.bool, device=high_states.device)
        source_positions = torch.zeros(
            batch, width, dtype=torch.long, device=high_states.device
        )
        target_positions = torch.zeros_like(source_positions)
        starts = torch.zeros_like(source_positions)
        ends_tensor = torch.zeros_like(source_positions)
        for row, row_groups in enumerate(groups):
            prompt = int(prompt_len[row])
            for column, (start, end) in enumerate(row_groups):
                source_position = prompt - 1 if start == 0 else prompt + start - 1
                target_position = prompt + end - 1
                length = end - start
                prev[row, column] = high_states[row, source_position]
                target[row, column] = high_targets[row, target_position]
                low_start[row, column] = seq["prev"][row, start]
                low_target[row, column] = seq["target"][row, end - 1]
                raw_actions[row, column, :length] = token_actions[row, start:end]
                raw_ids[row, column, :length] = seq["action_ids"][row, start:end]
                raw_valid[row, column, :length] = True
                valid[row, column] = True
                source_positions[row, column] = source_position
                target_positions[row, column] = target_position
                starts[row, column] = start
                ends_tensor[row, column] = end

        flat_prev = prev.reshape(batch * width, self.d_high)
        codes, extras = self.macro_action.training_code(
            raw_actions.reshape(batch * width, raw_width, -1), flat_prev,
            raw_valid.reshape(batch * width, raw_width),
        )
        codes = codes.reshape(batch, width, self.d_macro)
        extras = {
            key: value.reshape(batch, width, *value.shape[1:])
            for key, value in extras.items()
        }
        pred = self.high_predictor(prev, codes, valid)
        dense = MultilevelTokenHierarchyJEPA._dense_shifted(
            self.high_predictor, pred, target, codes, valid,
            self.high_dense_depth,
        )
        low_endpoint = low_start.new_zeros(low_start.shape)
        for row, row_groups in enumerate(groups):
            for column, (start, end) in enumerate(row_groups):
                low_endpoint[row, column] = self.low_predictor.rollout(
                    low_start[row:row + 1, column],
                    token_actions[row:row + 1, start:end],
                    state_history=seq["prev"][row:row + 1, :start + 1],
                    action_history=token_actions[row:row + 1, :start],
                )[0, -1]
        low_endpoint_high = self.low_to_high(low_endpoint)
        high_target_low = self.high_to_low(target)
        prompt_high = torch.stack([
            high_states[row, int(prompt_len[row]) - 1] for row in range(batch)
        ])
        scale = torch.tensor(
            [max(1, len(row)) for row in groups], device=high_states.device,
            dtype=high_states.dtype,
        )
        index = torch.arange(width, device=high_states.device).unsqueeze(0) + 1
        remaining = ((scale.unsqueeze(1) - index).clamp_min(0) / scale.unsqueeze(1))
        value = self.high_value(pred, prompt_high)
        macro_value = self.macro_value(prev, prompt_high[:, None].expand_as(prev), codes)
        support_pos = self.macro_support(prev, codes)
        support_neg = self.macro_support(prev, codes.roll(1, 0))
        reachability_logit = self.reachability(low_start, pred)
        diff = extras["macro_q_mu"].detach() - extras["macro_p_mu"]
        prior_nll = 0.5 * (
            math.log(2 * math.pi) + extras["macro_p_logvar"]
            + diff.square() * (-extras["macro_p_logvar"]).exp()
        ).mean(-1)
        return {
            "index": 0, "span": "sentence", "prev": prev,
            "target": target, "valid": valid, "codes": codes,
            "pred": pred, "dense_predictions": dense[0],
            "dense_targets": dense[1], "dense_masks": dense[2],
            "raw_action_windows": raw_actions, "raw_action_ids": raw_ids,
            "raw_action_valid": raw_valid, "start_positions": starts,
            "end_positions": ends_tensor, "source_positions": source_positions,
            "target_positions": target_positions, "low_start": low_start,
            "low_target": low_target,
            "low_endpoint": low_endpoint,
            "low_endpoint_high": low_endpoint_high,
            "recursive_low_endpoint": low_endpoint_high,
            "high_target_low": high_target_low,
            "remaining_target": remaining, "value": value,
            "macro_value": macro_value, "support_pos": support_pos,
            "support_neg": support_neg,
            "reachability_logit": reachability_logit,
            "prior_nll": prior_nll, "prompt_state": prompt_high, **extras,
        }

    def forward(self, tokens, prompt_len, sentence_ends):
        states = self.encoder(tokens)
        high_states = self.high_encoder(tokens)
        with torch.no_grad():
            targets = self.teacher(tokens)
            high_targets = self.high_teacher(tokens)
        seq = self._reasoning_sequences(states, targets, tokens, prompt_len)
        token_actions = self.token_action(seq["action_ids"])
        low_pred = self.low_predictor(seq["prev"], token_actions, seq["valid"])
        low_dense = MultilevelTokenHierarchyJEPA._dense_shifted(
            self.low_predictor, low_pred, seq["target"], token_actions,
            seq["valid"], self.low_dense_depth,
        )
        prior_input = seq["prev"].detach() if self.token_prior_detach_state else seq["prev"]
        token_prior_logits = self.token_prior(prior_input) if self.token_prior else None
        low_value = self.low_value(low_pred, seq["prompt_state"])
        length_scale = seq["lengths"].float().clamp_min(1)
        positions = torch.arange(seq["prev"].shape[1], device=tokens.device) + 1
        low_remaining = (
            seq["lengths"].unsqueeze(1).sub(positions).clamp_min(0).float()
            / length_scale.unsqueeze(1)
        )
        sentence = self._sentence_sequences(
            high_states, high_targets, seq, token_actions, prompt_len,
            sentence_ends,
        )
        prompt_high = sentence["prompt_state"]
        high_final_target = torch.stack([
            high_targets[row, tokens[row].ne(self.pad_id).sum() - 1]
            for row in range(tokens.shape[0])
        ])
        return {
            **seq, "states": states, "high_states": high_states,
            "high_targets": high_targets, "token_actions": token_actions,
            "token_prior_logits": token_prior_logits,
            "token_prior_rollout_logits": (), "low_pred": low_pred,
            "low_dense_predictions": low_dense[0],
            "low_dense_targets": low_dense[1], "low_dense_masks": low_dense[2],
            "low_value": low_value, "low_remaining_target": low_remaining,
            "goal_pred": self.low_goal_head(seq["prompt_state"]),
            "high_goal_pred": self.high_goal_head(prompt_high),
            "high_final_target": high_final_target,
            "sentence_level": sentence, "levels": [sentence],
        }

    def sentence_counterfactuals(
        self,
        out: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        prompt_len: torch.Tensor,
        k: int = 4,
        source: str = "in_batch",
        max_anchors: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate factual plus same-batch alternative complete sentences.

        Counterfactual outcomes are obtained by appending an observed sentence
        to the anchor prefix and running the EMA high encoder. The terminal
        reference state appears only in the advantage label. The value head
        itself receives ``(current state, prompt state, macro action)`` and is
        therefore deployable without access to the answer trace.
        """
        if source not in {"in_batch", "nearest"}:
            raise ValueError(f"unknown sentence counterfactual source: {source}")
        level = out["sentence_level"]
        anchors = level["valid"].nonzero(as_tuple=False)
        if max_anchors is not None and len(anchors) > int(max_anchors):
            select = torch.linspace(
                0, len(anchors) - 1, int(max_anchors), device=anchors.device
            ).long().unique()
            anchors = anchors[select]
        n = len(anchors)
        if n == 0:
            raise ValueError("counterfactual GAR requires a valid sentence")
        k = max(1, min(int(k), n))
        full_codes = level["codes"][level["valid"]]
        full_ids = level["raw_action_ids"][level["valid"]]
        full_valid = level["raw_action_valid"][level["valid"]]
        # Candidate zero must correspond to each selected factual anchor.
        flat_lookup = torch.full(
            level["valid"].shape, -1, dtype=torch.long,
            device=anchors.device,
        )
        flat_lookup[level["valid"]] = torch.arange(
            len(full_codes), device=anchors.device
        )
        anchor_bank_index = flat_lookup[anchors[:, 0], anchors[:, 1]]
        bank_codes, bank_ids, bank_valid = full_codes, full_ids, full_valid
        if source == "nearest" and len(bank_codes) > 1:
            distance = torch.cdist(
                bank_codes[anchor_bank_index].detach(), bank_codes.detach()
            )
            distance.scatter_(1, anchor_bank_index[:, None], torch.inf)
            neighbour_count = min(k - 1, len(bank_codes) - 1)
            nearest = distance.topk(neighbour_count, largest=False).indices
            factual = anchor_bank_index.unsqueeze(1)
            candidate_index = torch.cat([factual, nearest], 1)
        else:
            shifts = torch.arange(k, device=tokens.device).unsqueeze(0)
            candidate_index = (anchor_bank_index.unsqueeze(1) + shifts) % len(bank_codes)
        k_eff = candidate_index.shape[1]
        candidate_ids = bank_ids[candidate_index]
        candidate_valid = bank_valid[candidate_index]
        candidate_actions = self.token_action(candidate_ids)
        candidate_codes = self.macro_action.encoder(
            candidate_actions.reshape(n * k_eff, candidate_ids.shape[-1], -1),
            candidate_valid.reshape(n * k_eff, candidate_ids.shape[-1]),
        ).reshape(n, k_eff, self.d_macro)

        # Exact counterfactual target: causal high encoding of the actual
        # anchor prefix followed by the complete alternative sentence.
        sequences, sequence_lengths = [], []
        for anchor_index, (row_tensor, column_tensor) in enumerate(anchors):
            row, column = int(row_tensor), int(column_tensor)
            source_position = int(level["source_positions"][row, column])
            prefix = tokens[row, :source_position + 1]
            for candidate in range(k_eff):
                suffix = candidate_ids[anchor_index, candidate][
                    candidate_valid[anchor_index, candidate]
                ]
                sequence = torch.cat([prefix, suffix])
                sequences.append(sequence)
                sequence_lengths.append(len(sequence))
        max_length = max(sequence_lengths)
        counterfactual_tokens = tokens.new_full(
            (n * k_eff, max_length), self.pad_id
        )
        for index, sequence in enumerate(sequences):
            counterfactual_tokens[index, :len(sequence)] = sequence
        with torch.no_grad():
            encoded = self.high_teacher(counterfactual_tokens)
            exact_outcome = torch.stack([
                encoded[index, length - 1]
                for index, length in enumerate(sequence_lengths)
            ]).reshape(n, k_eff, self.d_high)

        # Candidate dynamics retain the observed sentence history. Only the
        # current macro action is swapped, and all candidates are evaluated in
        # one padded causal-transformer call.
        max_history = int(anchors[:, 1].max()) + 1
        history_state = level["prev"].new_zeros(
            n * k_eff, max_history, self.d_high
        )
        history_code = candidate_codes.new_zeros(
            n * k_eff, max_history, self.d_macro
        )
        history_valid = torch.zeros(
            n * k_eff, max_history, dtype=torch.bool, device=tokens.device
        )
        gather = []
        for anchor_index, (row_tensor, column_tensor) in enumerate(anchors):
            row, column = int(row_tensor), int(column_tensor)
            length = column + 1
            for candidate in range(k_eff):
                flat = anchor_index * k_eff + candidate
                history_state[flat, :length] = level["prev"][row, :length]
                if column:
                    history_code[flat, :column] = level["codes"][row, :column]
                history_code[flat, column] = candidate_codes[anchor_index, candidate]
                history_valid[flat, :length] = True
                gather.append(column)
        predicted_sequence = self.high_predictor(
            history_state, history_code, history_valid
        )
        gather_index = torch.tensor(gather, device=tokens.device)
        predicted_outcome = predicted_sequence[
            torch.arange(n * k_eff, device=tokens.device), gather_index
        ].reshape(n, k_eff, self.d_high)

        rows = anchors[:, 0]
        columns = anchors[:, 1]
        state = level["prev"][rows, columns]
        prompt = level["prompt_state"][rows]
        goal = out["high_final_target"][rows]
        value = self.macro_value(
            state[:, None].expand(-1, k_eff, -1),
            prompt[:, None].expand(-1, k_eff, -1), candidate_codes,
        )
        support = self.macro_support(
            state[:, None].expand(-1, k_eff, -1), candidate_codes
        )
        norm_state = torch.nn.functional.layer_norm(state, (self.d_high,))
        norm_goal = torch.nn.functional.layer_norm(goal, (self.d_high,))
        norm_outcome = torch.nn.functional.layer_norm(
            exact_outcome, (self.d_high,)
        )
        before = (norm_state - norm_goal).square().mean(-1)
        after = (norm_outcome - norm_goal[:, None]).square().mean(-1)
        advantage = before[:, None] - after
        return {
            "anchor_indices": anchors, "candidate_indices": candidate_index,
            "candidate_codes": candidate_codes,
            "candidate_ids": candidate_ids,
            "candidate_token_valid": candidate_valid,
            "predicted_outcome": predicted_outcome,
            "exact_outcome": exact_outcome, "advantage_target": advantage,
            "value": value, "support": support,
            "candidate_valid": torch.ones_like(advantage, dtype=torch.bool),
        }
