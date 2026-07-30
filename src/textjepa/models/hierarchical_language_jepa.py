"""Hierarchical predictive states over frozen language-model hidden states.

The module deliberately does not own a language model.  Callers obtain exact
hidden states and executable token proposals from a frozen causal LM, then
pass those hidden states here.  This keeps the JEPA learner, branch
generation, and exact re-encoding as separate operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from textjepa.models.ema import EMATeacher

HIERARCHICAL_LANGUAGE_ARCHITECTURE = (
    "nested_e0_e0_to_1_v2_window_exact"
)


@dataclass(frozen=True)
class HierarchicalLanguageJEPAConfig:
    d_backbone: int
    vocab_size: int
    pad_id: int = 0
    d_token: int = 256
    d_sentence: int = 128
    d_action: int = 32
    d_task: int = 256
    predictor_width: int = 512
    token_layers: int = 4
    sentence_layers: int = 3
    n_heads: int = 8
    token_context: int = 64
    sentence_context: int = 32
    cache_dropout: float = 0.0
    max_span: int = 128
    enable_macro_actions: bool = False
    enable_value: bool = False

    def __post_init__(self) -> None:
        positive = {
            "d_backbone": self.d_backbone, "vocab_size": self.vocab_size,
            "d_token": self.d_token, "d_sentence": self.d_sentence,
            "d_action": self.d_action, "d_task": self.d_task,
            "predictor_width": self.predictor_width,
            "token_layers": self.token_layers,
            "sentence_layers": self.sentence_layers,
            "n_heads": self.n_heads, "token_context": self.token_context,
            "sentence_context": self.sentence_context,
            "max_span": self.max_span,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if not 0 <= self.pad_id < self.vocab_size:
            raise ValueError("pad_id must lie inside the vocabulary")
        if not 0 <= self.cache_dropout < 1:
            raise ValueError("cache_dropout must be in [0, 1)")


class PlanningEncoder(nn.Module):
    """RMSNorm -> Linear -> SiLU -> Linear planning projection."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(d_in),
            nn.Linear(d_in, 2 * d_out),
            nn.SiLU(),
            nn.Linear(2 * d_out, d_out),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class ContextualControlledPredictor(nn.Module):
    """Bounded causal state-action predictor with residual state prediction.

    A rollout carries its own ``state_history`` and ``action_history``.
    Context regularization can truncate the prefix and independently drop
    cached pairs, while always retaining the current pair.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        width: int,
        layers: int,
        heads: int,
        max_context: int,
        cache_dropout: float = 0.0,
    ):
        super().__init__()
        if max_context < 1:
            raise ValueError("max_context must be positive")
        if not 0.0 <= cache_dropout < 1.0:
            raise ValueError("cache_dropout must be in [0, 1)")
        self.d_state = d_state
        self.d_action = d_action
        self.heads = int(heads)
        self.max_context = int(max_context)
        self.cache_dropout = float(cache_dropout)
        self.inp = nn.Linear(d_state + d_action, width)
        self.pos = nn.Parameter(torch.empty(1, max_context, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.RMSNorm(width)
        self.delta = nn.Linear(width + d_action, d_state)

    def _bounded(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if states.shape[:-1] != actions.shape[:-1]:
            raise ValueError("state and action histories must align")
        if valid.shape != states.shape[:-1]:
            raise ValueError("valid mask must match the history")
        length = states.shape[1]
        keep = min(length, self.max_context)
        states = states[:, -keep:]
        actions = actions[:, -keep:]
        valid = valid[:, -keep:].clone()
        valid[:, -1] = True
        return states, actions, valid

    def _positions(self, length: int) -> torch.Tensor:
        if length <= self.pos.shape[1]:
            return self.pos[:, :length]
        return torch.nn.functional.interpolate(
            self.pos.transpose(1, 2),
            size=length,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _predict_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
        random_truncation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = states.shape[:2]
        windows = self.max_context
        context_chunks = []
        # Window expansion makes dense and rollout semantics identical. Bound
        # peak memory by processing query positions in small rectangular
        # chunks rather than materializing B*L*C transformer items at once.
        for query_begin in range(0, length, 32):
            query_end = min(length, query_begin + 32)
            queries = query_end - query_begin
            chunk_windows = min(windows, query_end)
            key = torch.arange(chunk_windows, device=states.device)
            causal = key[None, :] > key[:, None]
            state_window = states.new_zeros(
                batch, queries, chunk_windows, states.shape[-1]
            )
            action_window = actions.new_zeros(
                batch, queries, chunk_windows, actions.shape[-1]
            )
            window_valid = torch.zeros(
                batch, queries, chunk_windows,
                dtype=torch.bool, device=states.device,
            )
            query_position = torch.zeros(
                batch, queries, dtype=torch.long, device=states.device
            )
            for local, query in enumerate(range(query_begin, query_end)):
                available = min(query + 1, chunk_windows)
                if random_truncation and self.training:
                    available = int(torch.randint(
                        1, available + 1, (), device=states.device
                    ))
                start = query + 1 - available
                state_window[:, local, :available] = states[
                    :, start:query + 1
                ]
                action_window[:, local, :available] = actions[
                    :, start:query + 1
                ]
                window_valid[:, local, :available] = valid[
                    :, start:query + 1
                ]
                query_position[:, local] = available - 1
            flat_states = state_window.flatten(0, 1)
            flat_actions = action_window.flatten(0, 1)
            flat_valid = window_valid.flatten(0, 1)
            previous_actions = torch.zeros_like(flat_actions)
            previous_actions[:, 1:] = flat_actions[:, :-1]
            item = self.inp(torch.cat([flat_states, previous_actions], -1))
            rows_in_chunk = batch * queries
            attention_mask = causal[None].expand(
                rows_in_chunk, chunk_windows, chunk_windows
            ).clone()
            attention_mask |= ~flat_valid[:, None, :]
            if self.training and self.cache_dropout:
                prior = key[None, :] < key[:, None]
                attention_mask |= (
                    torch.rand(
                        rows_in_chunk, chunk_windows, chunk_windows,
                        device=states.device,
                    ).lt(self.cache_dropout)
                    & prior
                )
            diagonal = torch.arange(chunk_windows, device=states.device)
            attention_mask[:, diagonal, diagonal] = False
            encoded = self.norm(self.blocks(
                item,
                mask=attention_mask.repeat_interleave(self.heads, 0),
            ))
            flat_query = query_position.reshape(-1)
            rows = torch.arange(rows_in_chunk, device=states.device)
            chunk_context = encoded[rows, flat_query].reshape(
                batch, queries, -1
            )
            context_chunks.append(chunk_context)
        context = torch.cat(context_chunks, 1)
        transition = self.delta(torch.cat([context, actions], -1))
        return states + transition, context

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor | None = None,
        *,
        random_truncation: bool = False,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        squeeze = states.ndim == 2
        if squeeze:
            states, actions = states[:, None], actions[:, None]
        if valid is None:
            valid = torch.ones(
                states.shape[:2], dtype=torch.bool, device=states.device
            )
        prediction, context = self._predict_sequence(
            states, actions, valid, random_truncation
        )
        if squeeze:
            prediction, context = prediction[:, 0], context[:, 0]
        return (prediction, context) if return_context else prediction

    def rollout(
        self,
        start: torch.Tensor,
        actions: torch.Tensor,
        *,
        state_history: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll out independent branches and return states and final context."""
        if actions.ndim != 3:
            raise ValueError("actions must be [branches, horizon, d_action]")
        branches, horizon, _ = actions.shape
        if horizon < 1:
            raise ValueError("rollout horizon must be positive")
        if state_history is None:
            if start.ndim == 1:
                initial = start[None].expand(branches, -1)
            elif start.shape == (1, self.d_state):
                initial = start.expand(branches, -1)
            elif start.shape == (branches, self.d_state):
                initial = start
            else:
                raise ValueError("start must be [D], [1,D], or [branches,D]")
            states = initial[:, None]
            prior_actions = actions[:, :0]
        else:
            states = state_history
            if states.shape[0] == 1 and branches != 1:
                states = states.expand(branches, -1, -1)
            prior_actions = (
                actions[:, :0] if action_history is None else action_history
            )
            if prior_actions.shape[1] != states.shape[1] - 1:
                raise ValueError(
                    "action_history must contain one fewer item than "
                    "state_history"
                )
            if prior_actions.shape[0] == 1 and branches != 1:
                prior_actions = prior_actions.expand(branches, -1, -1)
        outputs, final_context = [], None
        for index in range(horizon):
            prior_actions = torch.cat(
                [prior_actions, actions[:, index:index + 1]], 1
            )
            valid = torch.ones(
                states.shape[:2], dtype=torch.bool, device=states.device
            )
            states, prior_actions, valid = self._bounded(
                states, prior_actions, valid
            )
            prediction, context = self(
                states, prior_actions, valid, return_context=True
            )
            current = prediction[:, -1]
            final_context = context[:, -1]
            outputs.append(current)
            states = torch.cat([states, current[:, None]], 1)
        return torch.stack(outputs, 1), final_context


class DeterministicSentenceActionEncoder(nn.Module):
    """Encode an observed complete reasoning step without state leakage."""

    def __init__(
        self, vocab_size: int, pad_id: int, d_action: int, max_span: int
    ):
        super().__init__()
        width = max(64, 4 * d_action)
        self.max_span = int(max_span)
        self.embedding = nn.Embedding(
            vocab_size, width, padding_idx=pad_id
        )
        self.position = nn.Parameter(torch.empty(1, max_span, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width, 4, 4 * width, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(
            nn.RMSNorm(width),
            nn.Linear(width, 2 * d_action),
            nn.SiLU(),
            nn.Linear(2 * d_action, d_action),
        )

    def forward(
        self,
        span_token_ids: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        if span_mask.shape != span_token_ids.shape:
            raise ValueError("span mask shape is invalid")
        if span_token_ids.shape[-1] > self.max_span:
            raise ValueError("sentence action exceeds maximum span")
        shape = span_token_ids.shape[:-1]
        length = span_token_ids.shape[-1]
        ids = span_token_ids.reshape(-1, length)
        valid = span_mask.reshape(-1, length)
        safe = valid.clone()
        safe[~safe.any(-1), 0] = True
        hidden = self.embedding(ids) + self.position[:, :length]
        hidden = self.blocks(hidden, src_key_padding_mask=~safe)
        weight = valid.to(hidden.dtype)
        pooled = (hidden * weight[..., None]).sum(1) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1)
        return self.out(pooled).reshape(*shape, -1)


class GaussianMacroActions(nn.Module):
    """Training posterior and causal prior for supported macro-actions."""

    def __init__(self, d_state: int, d_context: int, d_task: int, d_action: int):
        super().__init__()
        posterior_in = 2 * d_state + d_context + d_action
        prior_in = d_state + d_context + d_task
        self.posterior = nn.Sequential(
            nn.RMSNorm(posterior_in),
            nn.Linear(posterior_in, 4 * d_action),
            nn.SiLU(),
            nn.Linear(4 * d_action, 2 * d_action),
        )
        self.prior = nn.Sequential(
            nn.RMSNorm(prior_in),
            nn.Linear(prior_in, 4 * d_action),
            nn.SiLU(),
            nn.Linear(4 * d_action, 2 * d_action),
        )

    @staticmethod
    def _params(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = raw.chunk(2, -1)
        return mean, logvar.clamp(-8.0, 3.0)

    def posterior_params(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        observed: torch.Tensor,
        successor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params(self.posterior(
            torch.cat([state, context, observed, successor - state], -1)
        ))

    def prior_params(
        self, state: torch.Tensor, context: torch.Tensor, task: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._params(self.prior(torch.cat([state, context, task], -1)))

    @staticmethod
    def reparameterize(
        mean: torch.Tensor, logvar: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        noise = torch.randn_like(mean) if noise is None else noise
        return mean + noise * (0.5 * logvar).exp()


class TaskConditionedValue(nn.Module):
    """Unbudgeted ``V_eta(xi^1, q)``; no remaining-budget input exists."""

    def __init__(self, d_state: int, d_context: int, d_task: int):
        super().__init__()
        width = d_state + d_context + d_task
        self.net = nn.Sequential(
            nn.RMSNorm(width),
            nn.Linear(width, 2 * d_context),
            nn.SiLU(),
            nn.Linear(2 * d_context, 1),
        )

    def forward(
        self, state: torch.Tensor, context: torch.Tensor, task: torch.Tensor
    ) -> torch.Tensor:
        task = task.unsqueeze(-2).expand(
            *context.shape[:-1], task.shape[-1]
        ) if task.ndim < context.ndim else task
        return self.net(torch.cat([state, context, task], -1)).squeeze(-1)


class HierarchicalLanguageJEPA(nn.Module):
    """Token and sentence predictive spaces for frozen-LM hidden states."""

    def __init__(self, config: HierarchicalLanguageJEPAConfig):
        super().__init__()
        self.config = config
        # Canonical notation:
        # E_LM (external frozen backbone) -> E0 -> E0_to_1.
        self.e0 = PlanningEncoder(
            config.d_backbone, config.d_token
        )
        self.e0_target = EMATeacher(self.e0)
        self.e0_to_1 = nn.Sequential(
            nn.RMSNorm(config.d_token),
            nn.Linear(config.d_token, 2 * config.d_sentence),
            nn.SiLU(),
            nn.Linear(2 * config.d_sentence, config.d_sentence),
        )
        self.e0_to_1_target = EMATeacher(self.e0_to_1)
        self.token_action = nn.Embedding(config.vocab_size, config.d_token)
        self.p0 = ContextualControlledPredictor(
            config.d_token, config.d_token, config.predictor_width,
            config.token_layers, config.n_heads, config.token_context,
            config.cache_dropout,
        )
        self.a1 = DeterministicSentenceActionEncoder(
            config.vocab_size, config.pad_id, config.d_action, config.max_span
        )
        self.p1 = ContextualControlledPredictor(
            config.d_sentence, config.d_action, config.predictor_width,
            config.sentence_layers, config.n_heads, config.sentence_context,
            config.cache_dropout,
        )
        self.task_projection = nn.Sequential(
            nn.RMSNorm(config.d_backbone),
            nn.Linear(config.d_backbone, config.d_task),
        )
        self.pi1 = (
            GaussianMacroActions(
                config.d_sentence, config.predictor_width,
                config.d_task, config.d_action,
            )
            if config.enable_macro_actions else None
        )
        self.v = (
            TaskConditionedValue(
                config.d_sentence, config.predictor_width, config.d_task
            )
            if config.enable_value else None
        )

    # Compatibility handles for pre-migration callers. Canonical code uses
    # e0, e0_to_1, p0, a1, p1, pi1, and v. These properties never register a
    # second representation path.
    @property
    def token_encoder(self):
        return self.e0

    @property
    def token_projector(self):
        return self.e0

    @property
    def token_target(self):
        return self.e0_target

    @property
    def sentence_encoder(self):
        return self.e0_to_1

    @property
    def fine_to_coarse(self):
        """Deprecated name for the one authoritative E0_to_1 encoder."""
        return self.e0_to_1

    @property
    def token_predictor(self):
        return self.p0

    @property
    def sentence_predictor(self):
        return self.p1

    @property
    def sentence_action(self):
        return self.a1

    @property
    def macro_actions(self):
        return self.pi1

    @property
    def value(self):
        return self.v

    @torch.no_grad()
    def update_targets(self, momentum: float) -> None:
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be in [0, 1]")
        self.e0_target.update(self.e0, momentum)
        self.e0_to_1_target.update(self.e0_to_1, momentum)

    def encode_token(
        self, hidden: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        return self.e0_target(hidden) if target else self.e0(hidden)

    def encode_sentence_from_token(
        self, token_state: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        encoder = self.e0_to_1_target if target else self.e0_to_1
        return encoder(token_state)

    def encode_sentence(
        self, hidden: torch.Tensor, *, target: bool = False
    ) -> torch.Tensor:
        """Strict nested tower; there is no direct h -> z1 planning encoder."""
        token = self.encode_token(hidden, target=target)
        return self.encode_sentence_from_token(token, target=target)

    def task_embedding(
        self, hidden: torch.Tensor, prompt_len: torch.Tensor
    ) -> torch.Tensor:
        if prompt_len.shape != (len(hidden),):
            raise ValueError("prompt_len must have one value per example")
        if bool((prompt_len < 1).any()) or bool(
            (prompt_len > hidden.shape[1]).any()
        ):
            raise ValueError("prompt_len lies outside the hidden sequence")
        rows = torch.arange(len(hidden), device=hidden.device)
        return self.task_projection(hidden[rows, prompt_len - 1])

    @staticmethod
    def _gather_boundaries(
        values: torch.Tensor,
        boundaries: torch.Tensor,
        *,
        prefix_lengths: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = boundaries.ge(0)
        positions = boundaries - 1 if prefix_lengths else boundaries
        valid &= positions.ge(0) & positions.lt(values.shape[1])
        safe = positions.clamp(0, values.shape[1] - 1)
        index = safe[..., None].expand(*safe.shape, values.shape[-1])
        return values.gather(1, index), valid

    def encode(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token = self.encode_token(hidden)
        sentence = self.encode_sentence_from_token(token)
        with torch.no_grad():
            token_target = self.encode_token(hidden, target=True)
            sentence_target = self.encode_sentence_from_token(
                token_target, target=True
            )
        return token, token_target, sentence, sentence_target

    @staticmethod
    def _validate_padding(
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        if attention_mask.shape != token_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must be boolean and match token_ids")
        if bool((~attention_mask[:, :-1] & attention_mask[:, 1:]).any()):
            raise ValueError("attention padding must be right-contiguous")
        if bool((token_ids[~attention_mask] != pad_id).any()):
            raise ValueError("padding positions must contain pad_id")
        return attention_mask.sum(-1).long()

    @staticmethod
    def _validate_boundaries(
        boundaries: torch.Tensor,
        valid_lengths: torch.Tensor,
        prompt_len: torch.Tensor | None,
        solution_end: torch.Tensor | None,
    ) -> torch.Tensor:
        boundary_mask = boundaries.ge(0)
        if bool((~boundary_mask[:, :-1] & boundary_mask[:, 1:]).any()):
            raise ValueError("boundary padding must be right-contiguous")
        for row in range(len(boundaries)):
            row_boundaries = boundaries[row, boundary_mask[row]]
            if len(row_boundaries) < 2:
                raise ValueError("each example needs at least two boundaries")
            if bool((row_boundaries[1:] <= row_boundaries[:-1]).any()):
                raise ValueError("valid boundaries must be strictly increasing")
            if int(row_boundaries[-1]) > int(valid_lengths[row]):
                raise ValueError("boundary reaches into padding")
            if prompt_len is not None:
                if int(row_boundaries[0]) != int(prompt_len[row]):
                    raise ValueError("first boundary must equal prompt_len")
                if int(row_boundaries[-1]) != int(solution_end[row]):
                    raise ValueError("last boundary must equal solution_end")
        return boundary_mask

    def token_forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        prompt_len: torch.Tensor | None = None,
        solution_end: torch.Tensor | None = None,
        random_context_truncation: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Dense canonical ``z_n --x_n--> z_{n+1}`` token transitions."""
        token = self.encode_token(hidden)
        with torch.no_grad():
            token_target = self.encode_token(hidden, target=True)
        if prompt_len is None:
            valid = token_ids[:, 1:].ne(self.config.pad_id)
            action_ids = token_ids[:, 1:]
            actions = self.token_action(action_ids)
            states, targets = token[:, :-1], token_target[:, 1:]
        else:
            if solution_end is None:
                raise ValueError("solution_end is required with prompt_len")
            widths = solution_end - prompt_len
            if bool((widths < 1).any()):
                raise ValueError("solution must contain at least one token")
            width = int(widths.max())
            batch = len(hidden)
            states = token.new_zeros(batch, width, self.config.d_token)
            targets = token.new_zeros(batch, width, self.config.d_token)
            action_ids = token_ids.new_full(
                (batch, width), self.config.pad_id
            )
            valid = torch.zeros(
                batch, width, dtype=torch.bool, device=hidden.device
            )
            for row in range(batch):
                start, end = int(prompt_len[row]), int(solution_end[row])
                count = end - start
                states[row, :count] = token[row, start - 1:end - 1]
                targets[row, :count] = token_target[row, start:end]
                action_ids[row, :count] = token_ids[row, start:end]
                valid[row, :count] = True
            actions = self.token_action(action_ids)
        prediction = self.token_predictor(
            states, actions, valid,
            random_truncation=random_context_truncation,
        )
        return {
            "token_states": states,
            "token_targets": targets,
            "token_predictions": prediction,
            "token_actions": actions,
            "token_action_ids": action_ids,
            "token_valid": valid,
        }

    def sentence_forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        boundaries: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        prompt_len: torch.Tensor | None = None,
        solution_end: torch.Tensor | None = None,
        random_context_truncation: bool = False,
    ) -> dict[str, torch.Tensor]:
        valid_lengths = attention_mask.sum(-1).long()
        boundary_valid = self._validate_boundaries(
            boundaries, valid_lengths, prompt_len, solution_end
        )
        prefix_lengths = prompt_len is not None
        token_all = self.encode_token(hidden)
        sentence_all = self.encode_sentence_from_token(token_all)
        with torch.no_grad():
            target_token_all = self.encode_token(hidden, target=True)
            sentence_target_all = self.encode_sentence_from_token(
                target_token_all, target=True
            )
        sentence, _ = self._gather_boundaries(
            sentence_all, boundaries, prefix_lengths=prefix_lengths
        )
        sentence_target, _ = self._gather_boundaries(
            sentence_target_all, boundaries, prefix_lengths=prefix_lengths
        )
        transition_valid = boundary_valid[:, :-1] & boundary_valid[:, 1:]
        batch, steps = transition_valid.shape
        span_token_ids = boundaries.new_full(
            (batch, steps, self.config.max_span), self.config.pad_id
        )
        span_mask = torch.zeros(
            batch, steps, self.config.max_span,
            dtype=torch.bool, device=hidden.device,
        )
        for row in range(batch):
            for step in range(steps):
                if not bool(transition_valid[row, step]):
                    continue
                start = int(boundaries[row, step])
                end = int(boundaries[row, step + 1])
                length = end - start
                if length > self.config.max_span:
                    raise ValueError(
                        f"reasoning span length {length} exceeds max_span"
                    )
                token_start = start if prefix_lengths else start + 1
                token_end = end if prefix_lengths else end + 1
                span_token_ids[row, step, :length] = token_ids[
                    row, token_start:token_end
                ]
                span_mask[row, step, :length] = True
        actions = self.a1(span_token_ids, span_mask)
        prediction, context = self.sentence_predictor(
            sentence[:, :-1], actions, transition_valid,
            random_truncation=random_context_truncation,
            return_context=True,
        )
        return {
            "sentence_states": sentence[:, :-1],
            "sentence_targets": sentence_target[:, 1:],
            "sentence_predictions": prediction,
            "sentence_actions": actions,
            "sentence_context": context,
            "sentence_valid": transition_valid,
            "coarse_at_boundaries": sentence_target,
            "boundary_valid": boundary_valid,
        }

    def dense_forward(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        boundaries: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        prompt_len: torch.Tensor | None = None,
        solution_end: torch.Tensor | None = None,
        include_sentence: bool = True,
        include_token_dynamics: bool = True,
        random_context_truncation: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute dense token and boundary-to-boundary sentence transitions.

        ``boundaries`` contains LM positions at the start and end of each
        reasoning operation, padded with ``-1``.  A row ``[3, 7, 11]`` yields
        spans ``(3, 7]`` and ``(7, 11]``.
        """
        if hidden.shape[:2] != token_ids.shape:
            raise ValueError("hidden states and token ids must align")
        if hidden.shape[-1] != self.config.d_backbone:
            raise ValueError("unexpected frozen-LM hidden width")
        attention_mask = (
            token_ids.ne(self.config.pad_id)
            if attention_mask is None else attention_mask
        )
        valid_lengths = self._validate_padding(
            token_ids, attention_mask, self.config.pad_id
        )
        if prompt_len is not None:
            if solution_end is None:
                raise ValueError("solution_end is required with prompt_len")
            if bool((solution_end > valid_lengths).any()):
                raise ValueError("solution_end reaches into padding")
        output = {}
        if include_token_dynamics:
            output.update(self.token_forward(
                hidden, token_ids, attention_mask=attention_mask,
                prompt_len=prompt_len, solution_end=solution_end,
                random_context_truncation=random_context_truncation,
            ))
        if not include_sentence:
            return output
        sentence_output = self.sentence_forward(
            hidden, token_ids, boundaries, attention_mask=attention_mask,
            prompt_len=prompt_len, solution_end=solution_end,
            random_context_truncation=random_context_truncation,
        )
        output.update(sentence_output)
        token = self.encode_token(hidden)
        output["fine_at_boundaries"] = self._gather_boundaries(
            token, boundaries, prefix_lengths=prompt_len is not None
        )[0]
        return output
