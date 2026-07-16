"""Multilevel causal token-to-span JEPA for action-free reasoning text."""

from __future__ import annotations

import math

import torch
from torch import nn

from textjepa.models.action import MacroActionModel
from textjepa.models.ema import EMATeacher
from textjepa.models.heads import MacroSupportHead, ValueHead
from textjepa.models.layers import build_causal_attention_mask, mlp
from textjepa.models.predictor import CausalHistoryPredictor
from textjepa.models.token_hierarchy import CausalTokenStateEncoder


class CausalLevelStateEncoder(nn.Module):
    """Causally re-encode a lower-level state path into a new state space."""

    def __init__(self, d_state, n_layers, n_heads, ff_mult, max_steps):
        super().__init__()
        self.n_heads = n_heads
        self.inp = nn.Linear(d_state, d_state)
        self.pos = nn.Parameter(torch.zeros(1, max_steps, d_state))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_state, n_heads, d_state * ff_mult, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_state)

    def _positions(self, length):
        if length <= self.pos.shape[1]:
            return self.pos[:, :length]
        return torch.nn.functional.interpolate(
            self.pos.transpose(1, 2), size=length, mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, states, valid):
        safe_valid = valid.clone()
        safe_valid[~safe_valid.any(1), 0] = True
        h = self.inp(states) + self._positions(states.shape[1])
        mask = build_causal_attention_mask(safe_valid, self.n_heads)
        return self.norm(self.blocks(h, mask=mask))


class TokenHierarchyLevel(nn.Module):
    def __init__(
        self,
        d_state: int,
        d_in_action: int,
        d_action: int,
        ratio: int,
        predictor_layers: int,
        n_heads: int,
        ff_mult: int,
        variational: bool,
        concat_width: int,
        max_steps: int,
        distinct_state_space: bool,
        state_encoder_layers: int,
    ):
        super().__init__()
        self.ratio = ratio
        self.state_encoder = (
            CausalLevelStateEncoder(
                d_state, state_encoder_layers, n_heads, ff_mult,
                max_steps=max_steps * ratio + 1,
            ) if distinct_state_space else None
        )
        self.state_teacher = (
            EMATeacher(self.state_encoder)
            if self.state_encoder is not None else None
        )
        self.action = MacroActionModel(
            d_in_action,
            d_state,
            d_action,
            ratio,
            kind="concat",
            variational=variational,
            concat_width=concat_width,
        )
        self.predictor = CausalHistoryPredictor(
            d_state,
            d_action,
            predictor_layers,
            n_heads,
            ff_mult,
            max_steps=max_steps,
            residual=False,
        )
        self.value = ValueHead(d_state)
        self.support = MacroSupportHead(d_state, d_action)


class MultilevelTokenHierarchyJEPA(nn.Module):
    """Token dynamics plus recursively constructed temporal abstractions.

    ``level_spans`` are absolute token strides. At every higher level, the
    action is an order-preserving projection of the complete sequence of
    lower-level actions in that span. Every transition model is causal.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        encoder_layers: int = 4,
        predictor_layers: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        max_len: int = 512,
        d_action: int = 64,
        level_spans: list[int] | tuple[int, ...] = (8,),
        level_dims: list[int] | tuple[int, ...] = (32,),
        variational_levels: list[bool] | tuple[bool, ...] = (False,),
        phase_augmented_levels: list[bool] | tuple[bool, ...] = (False,),
        concat_width: int = 8,
        low_dense_depth: int = 1,
        high_dense_depth: int = 1,
        use_token_prior: bool = False,
        token_prior_hidden: int = 0,
        token_prior_detach_state: bool = False,
        distinct_level_states: bool = False,
        level_state_encoder_layers: int = 2,
    ):
        super().__init__()
        spans = tuple(int(x) for x in level_spans)
        dims = tuple(int(x) for x in level_dims)
        variational = tuple(bool(x) for x in variational_levels)
        phase_augmented = tuple(bool(x) for x in phase_augmented_levels)
        if not spans or len(spans) != len(dims):
            raise ValueError("level_spans and level_dims must have equal nonzero length")
        if len(variational) == 1:
            variational = variational * len(spans)
        if len(variational) != len(spans):
            raise ValueError("variational_levels must have length one or n_levels")
        if len(phase_augmented) == 1:
            phase_augmented = phase_augmented * len(spans)
        if len(phase_augmented) != len(spans):
            raise ValueError(
                "phase_augmented_levels must have length one or n_levels"
            )
        previous = 1
        for span in spans:
            if span <= previous or span % previous:
                raise ValueError("each level span must be increasing and divisible by the previous")
            previous = span

        self.pad_id = pad_id
        self.d_model = d_model
        self.d_action = d_action
        self.level_spans = spans
        self.level_dims = dims
        self.phase_augmented_levels = phase_augmented
        self.low_dense_depth = max(1, int(low_dense_depth))
        self.high_dense_depth = max(1, int(high_dense_depth))
        self.token_prior_detach_state = bool(token_prior_detach_state)
        self.distinct_level_states = bool(distinct_level_states)
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
            prior_layers = [nn.LayerNorm(d_model)]
            if token_prior_hidden > 0:
                prior_layers.extend([
                    nn.Linear(d_model, token_prior_hidden),
                    nn.GELU(),
                    nn.Linear(token_prior_hidden, vocab_size),
                ])
            else:
                prior_layers.append(nn.Linear(d_model, vocab_size))
            self.token_prior = nn.Sequential(*prior_layers)
        else:
            self.token_prior = None
        levels = []
        previous_span, previous_dim = 1, d_action
        for span, dim, is_variational in zip(spans, dims, variational):
            levels.append(TokenHierarchyLevel(
                d_state=d_model,
                d_in_action=previous_dim,
                d_action=dim,
                ratio=span // previous_span,
                predictor_layers=predictor_layers,
                n_heads=n_heads,
                ff_mult=ff_mult,
                variational=is_variational,
                concat_width=concat_width,
                max_steps=max(8, max_len // span),
                distinct_state_space=self.distinct_level_states,
                state_encoder_layers=level_state_encoder_layers,
            ))
            previous_span, previous_dim = span, dim
        self.levels = nn.ModuleList(levels)
        self.goal_head = mlp([d_model, 2 * d_model], d_model)
        self.low_value = ValueHead(d_model)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        self.teacher.update(self.encoder, momentum)
        for level in self.levels:
            if level.state_teacher is not None:
                level.state_teacher.update(level.state_encoder, momentum)

    def encode_prefix(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)[:, -1]

    def lift_state_path(
        self, state_path: torch.Tensor, through_level: int | None = None,
        teacher: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Lift an observed base-state history (optionally ending in a goal).

        Planning can append the encoded terminal goal to the observed base
        history, call this method, and compare a level's predicted waypoint to
        the last lifted goal state.  The causal encoders see no states after
        the position they encode.  The final path element is retained even
        when it is not exactly on a fixed-span boundary.
        """
        if state_path.dim() != 3:
            raise ValueError("state_path must have shape [batch, time, state]")
        limit = len(self.levels) if through_level is None else through_level + 1
        if not 1 <= limit <= len(self.levels):
            raise ValueError("through_level is outside the configured hierarchy")
        current = state_path
        outputs = []
        previous_span = 1
        for index, (span, level) in enumerate(
            zip(self.level_spans[:limit], self.levels[:limit])
        ):
            valid = torch.ones(
                current.shape[:2], dtype=torch.bool, device=current.device
            )
            encoder = (
                level.state_teacher if teacher else level.state_encoder
            )
            encoded = encoder(current, valid) if encoder is not None else current
            outputs.append(encoded)
            ratio = span // previous_span
            indices = list(range(0, encoded.shape[1], ratio))
            if indices[-1] != encoded.shape[1] - 1:
                indices.append(encoded.shape[1] - 1)
            current = encoded[:, indices]
            previous_span = span
        return tuple(outputs)

    def _reasoning_sequences(
        self,
        states: torch.Tensor,
        targets: torch.Tensor,
        tokens: torch.Tensor,
        prompt_len: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, _, dim = states.shape
        lengths = (tokens.ne(self.pad_id).sum(1) - prompt_len).clamp_min(1)
        width = int(lengths.max())
        prev = states.new_zeros(batch, width, dim)
        target = states.new_zeros(batch, width, dim)
        ids = tokens.new_full((batch, width), self.pad_id)
        valid = torch.zeros(batch, width, dtype=torch.bool, device=tokens.device)
        prompt_state, prompt_target, final_target = [], [], []
        online_target = states.new_zeros(batch, width, dim)
        for b in range(batch):
            p, n = int(prompt_len[b]), int(lengths[b])
            prev[b, :n] = states[b, p - 1:p - 1 + n]
            target[b, :n] = targets[b, p:p + n]
            online_target[b, :n] = states[b, p:p + n]
            ids[b, :n] = tokens[b, p:p + n]
            valid[b, :n] = True
            prompt_state.append(states[b, p - 1])
            prompt_target.append(targets[b, p - 1])
            final_target.append(targets[b, p + n - 1])
        return {
            "prev": prev,
            "target": target,
            "online_target": online_target,
            "action_ids": ids,
            "valid": valid,
            "lengths": lengths,
            "prompt_state": torch.stack(prompt_state),
            "prompt_target": torch.stack(prompt_target),
            "final_target": torch.stack(final_target),
        }

    @staticmethod
    def _dense_shifted(
        predictor: nn.Module,
        first: torch.Tensor,
        targets: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
        depth: int,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        predictions, shifted_targets, masks = [first], [targets], [valid]
        cur = first
        for horizon in range(2, min(depth, actions.shape[1]) + 1):
            cur = predictor(
                cur[:, :-1], actions[:, horizon - 1:], valid[:, horizon - 1:]
            )
            predictions.append(cur)
            shifted_targets.append(targets[:, horizon - 1:])
            masks.append(valid[:, horizon - 1:])
        return tuple(predictions), tuple(shifted_targets), tuple(masks)

    def _raw_recursive_endpoint(
        self,
        sequence_states: torch.Tensor,
        token_actions: torch.Tensor,
        macro_prev: torch.Tensor,
        raw_windows: torch.Tensor,
        valid: torch.Tensor,
        span: int,
        phase_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Execute each primitive chunk with its complete causal history."""
        endpoint = macro_prev.new_zeros(macro_prev.shape)
        for window_index in range(raw_windows.shape[1]):
            rows = valid[:, window_index]
            if not rows.any():
                continue
            token_start = int(phase_offsets[rows][0]) + window_index * span
            # Rows in a batch can use different random phases. Group by
            # phase so every causal history has the correct common length.
            if not bool((phase_offsets[rows] == phase_offsets[rows][0]).all()):
                for phase in phase_offsets[rows].unique():
                    phase_rows = rows & phase_offsets.eq(phase)
                    start_at = int(phase) + window_index * span
                    endpoint[phase_rows, window_index] = self.low_predictor.rollout(
                        macro_prev[phase_rows, window_index],
                        raw_windows[phase_rows, window_index],
                        state_history=sequence_states[phase_rows, :start_at + 1],
                        action_history=token_actions[phase_rows, :start_at],
                    )[:, -1]
                continue
            endpoint[rows, window_index] = self.low_predictor.rollout(
                macro_prev[rows, window_index],
                raw_windows[rows, window_index],
                state_history=sequence_states[rows, :token_start + 1],
                action_history=token_actions[rows, :token_start],
            )[:, -1]
        return endpoint

    def forward(self, tokens: torch.Tensor, prompt_len: torch.Tensor) -> dict:
        states = self.encoder(tokens)
        with torch.no_grad():
            targets = self.teacher(tokens)
        seq = self._reasoning_sequences(states, targets, tokens, prompt_len)
        token_actions = self.token_action(seq["action_ids"])
        prior_states = (
            seq["prev"].detach() if self.token_prior_detach_state else seq["prev"]
        )
        token_prior_logits = (
            self.token_prior(prior_states) if self.token_prior is not None else None
        )
        low_pred = self.low_predictor(seq["prev"], token_actions, seq["valid"])
        low_dense = self._dense_shifted(
            self.low_predictor, low_pred, seq["target"], token_actions,
            seq["valid"], self.low_dense_depth,
        )
        token_prior_rollout_logits = []
        if self.token_prior is not None:
            for prediction in low_dense[0]:
                if prediction.shape[1] <= 1:
                    break
                predicted_state = prediction[:, :-1]
                if self.token_prior_detach_state:
                    predicted_state = predicted_state.detach()
                token_prior_rollout_logits.append(
                    self.token_prior(predicted_state)
                )
        length_scale = seq["lengths"].float().clamp_min(1)
        position = torch.arange(token_actions.shape[1], device=tokens.device) + 1
        low_remaining = (
            (seq["lengths"].unsqueeze(1) - position).clamp_min(0).float()
            / length_scale.unsqueeze(1)
        )
        low_value = self.low_value(low_pred, seq["prompt_state"])
        goal_pred = self.goal_head(seq["prompt_state"])

        batch = tokens.shape[0]
        source_actions = token_actions
        source_stride = 1
        source_counts = seq["lengths"]
        source_width = token_actions.shape[1] + 1
        source_online_path = states.new_zeros(batch, source_width, self.d_model)
        source_target_path = states.new_zeros(batch, source_width, self.d_model)
        source_online_path[:, 0] = seq["prompt_state"]
        source_target_path[:, 0] = seq["prompt_target"]
        source_online_path[:, 1:] = seq["online_target"]
        source_target_path[:, 1:] = seq["target"]
        source_phase_offsets = torch.zeros(
            batch, dtype=torch.long, device=tokens.device
        )
        level_outputs = []
        for level_index, (span, module) in enumerate(zip(self.level_spans, self.levels)):
            ratio = span // source_stride
            if self.training and self.phase_augmented_levels[level_index]:
                phase_units = torch.randint(
                    ratio, (batch,), device=tokens.device
                )
            else:
                phase_units = torch.zeros(batch, dtype=torch.long, device=tokens.device)
            # Consume the actual valid macro grid produced by the lower
            # level. Absolute phases accumulate through the hierarchy.
            phase_offsets = source_phase_offsets + phase_units * source_stride
            counts = torch.div(
                (source_counts - phase_units).clamp_min(0),
                ratio,
                rounding_mode="floor",
            )
            width = max(1, int(counts.max()))
            prev = states.new_zeros(batch, width, self.d_model)
            target = states.new_zeros(batch, width, self.d_model)
            windows = source_actions.new_zeros(
                batch, width, ratio, source_actions.shape[-1]
            )
            raw_windows = token_actions.new_zeros(
                batch, width, span, token_actions.shape[-1]
            )
            raw_ids = tokens.new_full((batch, width, span), self.pad_id)
            valid = torch.zeros(batch, width, dtype=torch.bool, device=tokens.device)
            end_positions = tokens.new_zeros(batch, width)
            path_valid = (
                torch.arange(source_online_path.shape[1], device=tokens.device)
                .unsqueeze(0) <= source_counts.unsqueeze(1)
            )
            if module.state_encoder is not None:
                encoded_online_path = module.state_encoder(
                    source_online_path, path_valid
                )
                with torch.no_grad():
                    encoded_target_path = module.state_teacher(
                        source_target_path, path_valid
                    )
            else:
                encoded_online_path = source_online_path
                encoded_target_path = source_target_path
            for b in range(batch):
                count = int(counts[b])
                for j in range(count):
                    token_start = int(phase_offsets[b]) + j * span
                    action_start = int(phase_units[b]) + j * ratio
                    prev[b, j] = encoded_online_path[b, action_start]
                    target[b, j] = encoded_target_path[b, action_start + ratio]
                    windows[b, j] = source_actions[b, action_start:action_start + ratio]
                    raw_windows[b, j] = token_actions[b, token_start:token_start + span]
                    raw_ids[b, j] = seq["action_ids"][b, token_start:token_start + span]
                    valid[b, j] = True
                    end_positions[b, j] = token_start + span
            flat_prev = prev.reshape(batch * width, -1)
            code, extras = module.action.training_code(
                windows.reshape(batch * width, ratio, -1), flat_prev
            )
            code = code.reshape(batch, width, -1)
            extras = {
                name: value.reshape(batch, width, *value.shape[1:])
                for name, value in extras.items()
            }
            pred = module.predictor(prev, code, valid)
            dense = self._dense_shifted(
                module.predictor, pred, target, code, valid,
                self.high_dense_depth,
            )
            # This is a controller-derived target, not a gradient path into
            # the primitive dynamics. It must nevertheless retain the causal
            # history that the same predictor uses during planning.
            with torch.no_grad():
                endpoint = self._raw_recursive_endpoint(
                    seq["prev"], token_actions, prev, raw_windows, valid,
                    span, phase_offsets,
                )
            remaining = (
                (seq["lengths"].unsqueeze(1) - end_positions)
                .clamp_min(0).float() / length_scale.unsqueeze(1)
            )
            value = module.value(pred, seq["prompt_state"])
            support_pos = module.support(prev, code)
            shuffled = code.roll(1, 0)
            support_neg = module.support(prev, shuffled)
            if module.action.variational:
                prior = extras["macro_prior_loss"] / code.shape[-1]
            else:
                diff = extras["macro_q_mu"].detach() - extras["macro_p_mu"]
                prior = 0.5 * (
                    math.log(2 * math.pi)
                    + extras["macro_p_logvar"]
                    + diff.square() * (-extras["macro_p_logvar"]).exp()
                ).mean(-1)
            level_outputs.append({
                "index": level_index,
                "span": span,
                "phase_offsets": phase_offsets,
                "prev": prev,
                "target": target,
                "valid": valid,
                "action_windows": windows,
                "raw_action_windows": raw_windows,
                "raw_action_ids": raw_ids,
                "codes": code,
                "pred": pred,
                "dense_predictions": dense[0],
                "dense_targets": dense[1],
                "dense_masks": dense[2],
                "recursive_low_endpoint": endpoint,
                "remaining_target": remaining,
                "value": value,
                "support_pos": support_pos,
                "support_neg": support_neg,
                "prior_nll": prior,
                **extras,
            })
            next_width = max(1, int(counts.max()) + 1)
            next_online_path = states.new_zeros(
                batch, next_width, self.d_model
            )
            next_target_path = states.new_zeros(
                batch, next_width, self.d_model
            )
            for b in range(batch):
                count = int(counts[b])
                start = int(phase_units[b])
                indices = start + torch.arange(
                    count + 1, device=tokens.device
                ) * ratio
                next_online_path[b, :count + 1] = encoded_online_path[b, indices]
                next_target_path[b, :count + 1] = encoded_target_path[b, indices]
            source_actions, source_stride = code, span
            source_counts = counts
            source_phase_offsets = phase_offsets
            source_online_path = next_online_path
            source_target_path = next_target_path
        return {
            **seq,
            "states": states,
            "token_actions": token_actions,
            "token_prior_logits": token_prior_logits,
            "token_prior_rollout_logits": tuple(token_prior_rollout_logits),
            "low_pred": low_pred,
            "low_dense_predictions": low_dense[0],
            "low_dense_targets": low_dense[1],
            "low_dense_masks": low_dense[2],
            "low_value": low_value,
            "low_remaining_target": low_remaining,
            "goal_pred": goal_pred,
            "levels": level_outputs,
        }
