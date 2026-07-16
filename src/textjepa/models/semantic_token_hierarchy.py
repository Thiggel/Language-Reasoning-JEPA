"""Variable-duration phrase/sentence hierarchy in one causal state space."""

from __future__ import annotations

import math

import torch
from torch import nn

from textjepa.models.action import MacroActionModel
from textjepa.models.ema import EMATeacher
from textjepa.models.heads import MacroSupportHead, ValueHead
from textjepa.models.layers import mlp
from textjepa.models.predictor import CausalHistoryPredictor
from textjepa.models.token_hierarchy import CausalTokenStateEncoder
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


class SemanticHierarchyLevel(nn.Module):
    def __init__(self, d_state, d_in_action, d_action, predictor_layers,
                 n_heads, ff_mult, max_steps):
        super().__init__()
        self.action = MacroActionModel(
            d_in_action, d_state, d_action, span=64, kind="transformer",
            variational=False,
        )
        self.predictor = CausalHistoryPredictor(
            d_state, d_action, predictor_layers, n_heads, ff_mult,
            max_steps=max_steps, residual=False,
        )
        self.value = ValueHead(d_state)
        self.support = MacroSupportHead(d_state, d_action)


class SemanticBoundaryTokenHierarchyJEPA(nn.Module):
    """Token, phrase, and sentence transitions with variable action lengths."""

    def __init__(self, vocab_size, pad_id, d_model=256, encoder_layers=4,
                 predictor_layers=2, n_heads=8, ff_mult=4, max_len=768,
                 d_action=64, level_dims=(32, 16), low_dense_depth=2,
                 high_dense_depth=2, use_token_prior=False,
                 token_prior_hidden=0, token_prior_detach_state=False):
        super().__init__()
        if len(level_dims) != 2:
            raise ValueError("semantic hierarchy currently requires phrase and sentence levels")
        self.pad_id = pad_id
        self.d_model = d_model
        self.d_action = d_action
        self.level_dims = tuple(int(value) for value in level_dims)
        self.level_spans = ("phrase", "sentence")
        self.low_dense_depth = max(1, int(low_dense_depth))
        self.high_dense_depth = max(1, int(high_dense_depth))
        self.token_prior_detach_state = bool(token_prior_detach_state)
        self.encoder = CausalTokenStateEncoder(
            vocab_size, pad_id, d_model, encoder_layers, n_heads, ff_mult, max_len
        )
        self.teacher = EMATeacher(self.encoder)
        self.token_action = nn.Embedding(vocab_size, d_action, padding_idx=pad_id)
        self.low_predictor = CausalHistoryPredictor(
            d_model, d_action, predictor_layers, n_heads, ff_mult,
            max_steps=max_len, residual=False,
        )
        if use_token_prior:
            layers = [nn.LayerNorm(d_model)]
            if token_prior_hidden:
                layers.extend([
                    nn.Linear(d_model, token_prior_hidden), nn.GELU(),
                    nn.Linear(token_prior_hidden, vocab_size),
                ])
            else:
                layers.append(nn.Linear(d_model, vocab_size))
            self.token_prior = nn.Sequential(*layers)
        else:
            self.token_prior = None
        self.levels = nn.ModuleList([
            SemanticHierarchyLevel(
                d_model, d_action, self.level_dims[0], predictor_layers,
                n_heads, ff_mult, max_len,
            ),
            SemanticHierarchyLevel(
                d_model, self.level_dims[0], self.level_dims[1],
                predictor_layers, n_heads, ff_mult, max_len,
            ),
        ])
        self.goal_head = mlp([d_model, 2 * d_model], d_model)
        self.low_value = ValueHead(d_model)

    @torch.no_grad()
    def update_teacher(self, momentum):
        self.teacher.update(self.encoder, momentum)

    @staticmethod
    def _reasoning_sequences(states, targets, tokens, prompt_len, pad_id):
        batch, _, dim = states.shape
        lengths = (tokens.ne(pad_id).sum(1) - prompt_len).clamp_min(1)
        width = int(lengths.max())
        prev = states.new_zeros(batch, width, dim)
        target = states.new_zeros(batch, width, dim)
        ids = tokens.new_full((batch, width), pad_id)
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
        return dict(
            prev=prev, target=target, action_ids=ids, valid=valid,
            lengths=lengths, prompt_state=torch.stack(prompt_state),
            final_target=torch.stack(final_target),
        )

    def _segments(self, states, seq, token_actions, source_actions,
                  source_starts, source_ends, boundary_ends, module,
                  level_index):
        batch = states.shape[0]
        groups = []
        for row in range(batch):
            ends = [int(value) for value in boundary_ends[row] if int(value) > 0]
            row_groups, previous = [], 0
            for end in ends:
                indices = [
                    index for index in range(source_actions.shape[1])
                    if bool(source_ends[row, index] > previous)
                    and bool(source_ends[row, index] <= end)
                    and bool(source_starts[row, index] >= previous)
                ]
                if indices and end <= int(seq["lengths"][row]):
                    row_groups.append((previous, end, indices))
                previous = end
            groups.append(row_groups)
        width = max(1, max((len(row) for row in groups), default=0))
        action_width = max(
            1, max((len(indices) for row in groups for _, _, indices in row), default=0)
        )
        raw_width = max(
            1, max((end - start for row in groups for start, end, _ in row), default=0)
        )
        prev = states.new_zeros(batch, width, self.d_model)
        target = states.new_zeros(batch, width, self.d_model)
        windows = source_actions.new_zeros(
            batch, width, action_width, source_actions.shape[-1]
        )
        window_valid = torch.zeros(
            batch, width, action_width, dtype=torch.bool, device=states.device
        )
        raw_windows = token_actions.new_zeros(
            batch, width, raw_width, token_actions.shape[-1]
        )
        raw_ids = seq["action_ids"].new_full(
            (batch, width, raw_width), self.pad_id
        )
        raw_valid = torch.zeros(
            batch, width, raw_width, dtype=torch.bool, device=states.device
        )
        valid = torch.zeros(batch, width, dtype=torch.bool, device=states.device)
        starts = torch.zeros(batch, width, dtype=torch.long, device=states.device)
        ends = torch.zeros_like(starts)
        for row, row_groups in enumerate(groups):
            for column, (start, end, indices) in enumerate(row_groups):
                count = len(indices)
                length = end - start
                prev[row, column] = seq["prev"][row, start]
                target[row, column] = seq["target"][row, end - 1]
                windows[row, column, :count] = source_actions[row, indices]
                window_valid[row, column, :count] = True
                raw_windows[row, column, :length] = token_actions[row, start:end]
                raw_ids[row, column, :length] = seq["action_ids"][row, start:end]
                raw_valid[row, column, :length] = True
                valid[row, column] = True
                starts[row, column], ends[row, column] = start, end
        flat_prev = prev.reshape(batch * width, -1)
        code, extras = module.action.training_code(
            windows.reshape(batch * width, action_width, -1), flat_prev,
            window_valid.reshape(batch * width, action_width),
        )
        code = code.reshape(batch, width, -1)
        extras = {
            name: value.reshape(batch, width, *value.shape[1:])
            for name, value in extras.items()
        }
        prediction = module.predictor(prev, code, valid)
        dense = MultilevelTokenHierarchyJEPA._dense_shifted(
            module.predictor, prediction, target, code, valid,
            self.high_dense_depth,
        )
        endpoint = prev.new_zeros(prev.shape)
        with torch.no_grad():
            for row, row_groups in enumerate(groups):
                for column, (start, end, _) in enumerate(row_groups):
                    endpoint[row, column] = self.low_predictor.rollout(
                        prev[row:row + 1, column],
                        token_actions[row:row + 1, start:end],
                        state_history=seq["prev"][row:row + 1, :start + 1],
                        action_history=token_actions[row:row + 1, :start],
                    )[0, -1]
        scale = seq["lengths"].float().clamp_min(1)
        remaining = (
            (seq["lengths"].unsqueeze(1) - ends).clamp_min(0).float()
            / scale.unsqueeze(1)
        )
        value = module.value(prediction, seq["prompt_state"])
        support_pos = module.support(prev, code)
        support_neg = module.support(prev, code.roll(1, 0))
        diff = extras["macro_q_mu"].detach() - extras["macro_p_mu"]
        prior = 0.5 * (
            math.log(2 * math.pi) + extras["macro_p_logvar"]
            + diff.square() * (-extras["macro_p_logvar"]).exp()
        ).mean(-1)
        output = dict(
            index=level_index, span=self.level_spans[level_index], prev=prev,
            target=target, valid=valid, action_windows=windows,
            action_window_valid=window_valid, raw_action_windows=raw_windows,
            raw_action_ids=raw_ids, raw_action_valid=raw_valid,
            start_positions=starts, end_positions=ends, codes=code,
            pred=prediction, dense_predictions=dense[0], dense_targets=dense[1],
            dense_masks=dense[2], recursive_low_endpoint=endpoint,
            remaining_target=remaining, value=value, support_pos=support_pos,
            support_neg=support_neg, prior_nll=prior, **extras,
        )
        return output, starts, ends

    def forward(self, tokens, prompt_len, phrase_ends, sentence_ends):
        states = self.encoder(tokens)
        with torch.no_grad():
            targets = self.teacher(tokens)
        seq = self._reasoning_sequences(
            states, targets, tokens, prompt_len, self.pad_id
        )
        token_actions = self.token_action(seq["action_ids"])
        prior_states = seq["prev"].detach() if self.token_prior_detach_state else seq["prev"]
        token_prior_logits = self.token_prior(prior_states) if self.token_prior is not None else None
        low_pred = self.low_predictor(seq["prev"], token_actions, seq["valid"])
        low_dense = MultilevelTokenHierarchyJEPA._dense_shifted(
            self.low_predictor, low_pred, seq["target"], token_actions,
            seq["valid"], self.low_dense_depth,
        )
        rollout_logits = []
        if self.token_prior is not None:
            for prediction in low_dense[0]:
                if prediction.shape[1] <= 1:
                    break
                state = prediction[:, :-1]
                if self.token_prior_detach_state:
                    state = state.detach()
                rollout_logits.append(self.token_prior(state))
        length_scale = seq["lengths"].float().clamp_min(1)
        position = torch.arange(token_actions.shape[1], device=tokens.device) + 1
        low_remaining = (
            (seq["lengths"].unsqueeze(1) - position).clamp_min(0).float()
            / length_scale.unsqueeze(1)
        )
        low_value = self.low_value(low_pred, seq["prompt_state"])
        goal_pred = self.goal_head(seq["prompt_state"])
        token_starts = torch.arange(token_actions.shape[1], device=tokens.device)
        token_starts = token_starts[None].expand(tokens.shape[0], -1)
        token_ends = token_starts + 1
        phrase, phrase_starts, phrase_terminal = self._segments(
            states, seq, token_actions, token_actions, token_starts,
            token_ends, phrase_ends, self.levels[0], 0,
        )
        sentence, _, _ = self._segments(
            states, seq, token_actions, phrase["codes"], phrase_starts,
            phrase_terminal, sentence_ends, self.levels[1], 1,
        )
        return {
            **seq, "states": states, "token_actions": token_actions,
            "token_prior_logits": token_prior_logits,
            "token_prior_rollout_logits": tuple(rollout_logits),
            "low_pred": low_pred, "low_dense_predictions": low_dense[0],
            "low_dense_targets": low_dense[1], "low_dense_masks": low_dense[2],
            "low_value": low_value, "low_remaining_target": low_remaining,
            "goal_pred": goal_pred, "levels": [phrase, sentence],
        }
