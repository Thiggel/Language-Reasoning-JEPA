"""Single-level token-action JEPA in a causal sentence/prefix state space."""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.ema import EMATeacher
from textjepa.models.heads import MacroValueHead
from textjepa.models.layers import mlp
from textjepa.models.predictor import CausalHistoryPredictor
from textjepa.models.token_hierarchy import CausalTokenStateEncoder
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


class FlatSentenceJEPA(nn.Module):
    """Predict the next causal prefix representation from the next token.

    There is no macro action, subgoal bridge, or second planning level.  The
    causal encoder state at a period summarizes the completed sentence, while
    intermediate states summarize partial sentences in their full prompt and
    reasoning context.
    """

    def __init__(
        self, vocab_size: int, pad_id: int, d_state: int = 192,
        encoder_layers: int = 4, predictor_layers: int = 2,
        n_heads: int = 8, ff_mult: int = 4, max_len: int = 768,
        d_action: int = 64, dense_depth: int = 4,
        use_token_prior: bool = True, token_prior_hidden: int = 0,
        token_prior_detach_state: bool = False,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.vocab_size = int(vocab_size)
        self.d_state = int(d_state)
        self.d_action = int(d_action)
        self.dense_depth = max(1, int(dense_depth))
        self.token_prior_detach_state = bool(token_prior_detach_state)
        self.encoder = CausalTokenStateEncoder(
            vocab_size, pad_id, d_state, encoder_layers, n_heads, ff_mult,
            max_len,
        )
        self.teacher = EMATeacher(self.encoder)
        self.token_action = nn.Embedding(
            vocab_size, d_action, padding_idx=pad_id
        )
        self.predictor = CausalHistoryPredictor(
            d_state, d_action, predictor_layers, n_heads, ff_mult,
            max_steps=max_len, residual=False,
        )
        self.goal_head = mlp([d_state, 2 * d_state], d_state)
        self.token_value = MacroValueHead(d_state, d_action)
        if use_token_prior:
            layers: list[nn.Module] = [nn.LayerNorm(d_state)]
            if token_prior_hidden:
                layers.extend([
                    nn.Linear(d_state, token_prior_hidden), nn.GELU(),
                    nn.Linear(token_prior_hidden, vocab_size),
                ])
            else:
                layers.append(nn.Linear(d_state, vocab_size))
            self.token_prior = nn.Sequential(*layers)
        else:
            self.token_prior = None

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        self.teacher.update(self.encoder, momentum)

    def _sequences(self, states, targets, tokens, prompt_len):
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

    def forward(self, tokens, prompt_len):
        states = self.encoder(tokens)
        with torch.no_grad():
            targets = self.teacher(tokens)
        seq = self._sequences(states, targets, tokens, prompt_len)
        actions = self.token_action(seq["action_ids"])
        pred = self.predictor(seq["prev"], actions, seq["valid"])
        dense = MultilevelTokenHierarchyJEPA._dense_shifted(
            self.predictor, pred, seq["target"], actions, seq["valid"],
            self.dense_depth,
        )
        prior_state = (
            seq["prev"].detach()
            if self.token_prior_detach_state else seq["prev"]
        )
        logits = self.token_prior(prior_state) if self.token_prior else None
        prompt = seq["prompt_state"][:, None].expand_as(seq["prev"])
        value = self.token_value(seq["prev"], prompt, actions)
        return {
            **seq, "states": states, "targets": targets, "actions": actions,
            "pred": pred, "dense_predictions": dense[0],
            "dense_targets": dense[1], "dense_masks": dense[2],
            "token_prior_logits": logits, "value": value,
            "goal_pred": self.goal_head(seq["prompt_state"]),
        }

    def token_counterfactuals(
        self, out, tokens, prompt_len, k=8, max_anchors=16,
    ):
        """Exact one-token alternatives with geometry-derived advantages."""
        anchors = out["valid"].nonzero(as_tuple=False)
        if len(anchors) > int(max_anchors):
            pick = torch.linspace(
                0, len(anchors) - 1, int(max_anchors), device=tokens.device
            ).long().unique()
            anchors = anchors[pick]
        n = len(anchors)
        if not n:
            raise ValueError("counterfactual GAR requires a valid token")
        k = max(1, min(int(k), self.vocab_size - 1))
        factual = out["action_ids"][anchors[:, 0], anchors[:, 1]]
        offsets = torch.linspace(
            0, self.vocab_size - 1, k, device=tokens.device
        ).long()[None]
        candidate_ids = (factual[:, None] + offsets) % self.vocab_size
        candidate_ids[:, 0] = factual
        candidate_ids[candidate_ids.eq(self.pad_id)] = (
            candidate_ids[candidate_ids.eq(self.pad_id)] + 1
        ) % self.vocab_size
        candidate_ids[:, 0] = factual
        candidate_actions = self.token_action(candidate_ids)

        sequences, lengths = [], []
        for row_t, column_t in anchors:
            row, column = int(row_t), int(column_t)
            source = int(prompt_len[row]) - 1 + column
            prefix = tokens[row, :source + 1]
            for candidate in candidate_ids[len(lengths) // k]:
                sequence = torch.cat([prefix, candidate.view(1)])
                sequences.append(sequence)
                lengths.append(len(sequence))
        padded = tokens.new_full((n * k, max(lengths)), self.pad_id)
        for index, sequence in enumerate(sequences):
            padded[index, :len(sequence)] = sequence
        with torch.no_grad():
            encoded = self.teacher(padded)
            exact = torch.stack([
                encoded[index, length - 1] for index, length in enumerate(lengths)
            ]).reshape(n, k, self.d_state)

        max_history = int(anchors[:, 1].max()) + 1
        history_states = out["prev"].new_zeros(n * k, max_history, self.d_state)
        history_actions = candidate_actions.new_zeros(n * k, max_history, self.d_action)
        history_valid = torch.zeros(
            n * k, max_history, dtype=torch.bool, device=tokens.device
        )
        gather = []
        for anchor_index, (row_t, column_t) in enumerate(anchors):
            row, column = int(row_t), int(column_t)
            length = column + 1
            for candidate in range(k):
                flat = anchor_index * k + candidate
                history_states[flat, :length] = out["prev"][row, :length]
                if column:
                    history_actions[flat, :column] = out["actions"][row, :column]
                history_actions[flat, column] = candidate_actions[anchor_index, candidate]
                history_valid[flat, :length] = True
                gather.append(column)
        sequence_pred = self.predictor(history_states, history_actions, history_valid)
        gather = torch.tensor(gather, device=tokens.device)
        predicted = sequence_pred[
            torch.arange(n * k, device=tokens.device), gather
        ].reshape(n, k, self.d_state)
        rows, columns = anchors[:, 0], anchors[:, 1]
        state = out["prev"][rows, columns]
        prompt = out["prompt_state"][rows]
        goal = out["final_target"][rows]
        value = self.token_value(
            state[:, None].expand(-1, k, -1),
            prompt[:, None].expand(-1, k, -1), candidate_actions,
        )
        norm = lambda x: torch.nn.functional.layer_norm(x, (self.d_state,))
        before = (norm(state) - norm(goal)).square().mean(-1)
        after = (norm(exact) - norm(goal[:, None])).square().mean(-1)
        return {
            "anchor_indices": anchors, "candidate_ids": candidate_ids,
            "candidate_actions": candidate_actions,
            "exact_outcome": exact, "predicted_outcome": predicted,
            "advantage_target": before[:, None] - after, "value": value,
            "candidate_valid": torch.ones_like(value, dtype=torch.bool),
        }
