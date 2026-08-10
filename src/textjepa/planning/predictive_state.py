"""Goal geometry, value baselines, block beam search, and reward shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class _Projection(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(input_size),
            nn.Linear(input_size, 2 * output_size),
            nn.SiLU(),
            nn.Linear(2 * output_size, output_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class GoalDistanceModel(nn.Module):
    """Separate prompt-goal and trajectory-state encoders."""

    def __init__(self, state_size: int, geometry_size: int = 128):
        super().__init__()
        self.state_encoder = _Projection(state_size, geometry_size)
        self.goal_encoder = _Projection(state_size, geometry_size)

    def embed_goal(self, prompt_state: torch.Tensor) -> torch.Tensor:
        return self.goal_encoder(prompt_state)

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state)

    def forward(
        self, state: torch.Tensor, prompt_state: torch.Tensor
    ) -> torch.Tensor:
        goal = self.embed_goal(prompt_state)
        while goal.ndim < state.ndim:
            goal = goal.unsqueeze(-2)
        return (self.embed_state(state) - goal).norm(dim=-1)


class NeuralQuasimetric(nn.Module):
    """Directional nonnegative distance from separate source/goal features."""

    def __init__(self, state_size: int, geometry_size: int = 128):
        super().__init__()
        self.source = _Projection(state_size, geometry_size)
        self.goal = _Projection(state_size, geometry_size)
        self.weights = nn.Parameter(torch.zeros(geometry_size))

    def forward(
        self, state: torch.Tensor, prompt_state: torch.Tensor
    ) -> torch.Tensor:
        source = self.source(state)
        goal = self.goal(prompt_state)
        while goal.ndim < source.ndim:
            goal = goal.unsqueeze(-2)
        directional_gap = F.relu(goal - source)
        return (directional_gap * F.softplus(self.weights)).sum(-1)


def make_goal_distance_model(
    kind: str, state_size: int, geometry_size: int = 128
) -> nn.Module:
    if kind == "euclidean":
        return GoalDistanceModel(state_size, geometry_size)
    if kind == "quasimetric":
        return NeuralQuasimetric(state_size, geometry_size)
    raise ValueError(f"unknown goal-distance kind: {kind}")


class DirectValueModel(nn.Module):
    """Budget-conditioned success probability baseline."""

    def __init__(self, state_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(2 * state_size + 1),
            nn.Linear(2 * state_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        prompt_state: torch.Tensor,
        budget: torch.Tensor,
    ) -> torch.Tensor:
        while prompt_state.ndim < state.ndim:
            prompt_state = prompt_state.unsqueeze(-2)
        prompt_state = prompt_state.expand_as(state)
        while budget.ndim < state.ndim:
            budget = budget.unsqueeze(-1)
        budget = budget.expand(*state.shape[:-1], 1)
        value = torch.cat([state, prompt_state, budget.to(state.dtype)], dim=-1)
        return self.net(value).squeeze(-1)


@dataclass
class GoalLoss:
    total: torch.Tensor
    terminal: torch.Tensor
    time: torch.Tensor
    monotonicity: torch.Tensor
    negative: torch.Tensor


def _safe_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def goal_distance_loss(
    distances: torch.Tensor,
    *,
    valid: torch.Tensor,
    remaining_chunks: torch.Tensor,
    correct: torch.Tensor,
    terminal_indices: torch.Tensor,
    monotonic_margin: float = 0.25,
    negative_margin: float = 4.0,
) -> GoalLoss:
    """Temporal metric objective with correct and incorrect trajectories."""
    if distances.shape != valid.shape or distances.shape != remaining_chunks.shape:
        raise ValueError("distance targets and validity mask must align")
    batch = torch.arange(len(distances), device=distances.device)
    terminal_distance = distances[batch, terminal_indices]
    correct = correct.bool()
    terminal = _safe_mean(terminal_distance.square(), correct)
    correct_steps = valid.bool() & correct[:, None]
    time_each = F.huber_loss(
        distances, remaining_chunks.to(distances.dtype), reduction="none"
    )
    time = _safe_mean(time_each, correct_steps)
    pair_valid = correct_steps[:, :-1] & correct_steps[:, 1:]
    mono_each = F.relu(
        monotonic_margin + distances[:, 1:] - distances[:, :-1]
    )
    monotonicity = _safe_mean(mono_each, pair_valid)
    negative_mask = ~correct
    negative_each = F.relu(negative_margin - terminal_distance).square()
    negative = _safe_mean(negative_each, negative_mask)
    return GoalLoss(
        total=terminal + time + 0.5 * monotonicity + 0.5 * negative,
        terminal=terminal,
        time=time,
        monotonicity=monotonicity,
        negative=negative,
    )


def distance_value(distance: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.exp(-distance / temperature)


def hybrid_logit(
    direct_value_logit: torch.Tensor,
    distance: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    return direct_value_logit - beta * distance


def candidate_score(
    *,
    log_probability_sum: torch.Tensor,
    token_count: torch.Tensor,
    distance: torch.Tensor | None,
    value_logit: torch.Tensor | None,
    uncertainty: torch.Tensor | None = None,
    alpha: float = 0.2,
    beta: float = 1.0,
    eta: float = 1.0,
    kappa: float = 0.0,
) -> torch.Tensor:
    score = alpha * log_probability_sum / token_count.clamp_min(1)
    if distance is not None:
        score = score - beta * distance
    if value_logit is not None:
        score = score + eta * value_logit
    if uncertainty is not None:
        score = score - kappa * uncertainty
    return score


@dataclass
class BeamState:
    tokens: torch.Tensor
    latent: torch.Tensor
    log_probability_sum: torch.Tensor
    score: torch.Tensor
    root_ids: torch.Tensor


def block_beam_search(
    *,
    initial_latent: torch.Tensor,
    initial_tokens: torch.Tensor,
    generate_candidates: Callable[
        [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
    ],
    advance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    score_leaves: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    beam_width: int = 4,
    candidates_per_beam: int = 4,
    chunk_length: int = 8,
    depth: int = 4,
) -> BeamState:
    """Batched block beam independent of exact versus predicted transitions.

    Callbacks make the exact-state gate explicit: `advance` is either a full-LM
    replay or a recurrent jump, while candidate generation and frozen scoring
    remain identical.
    """
    if initial_latent.ndim != 2 or initial_tokens.ndim != 2:
        raise ValueError("initial beam tensors must be batch-major")
    if initial_latent.shape[0] != initial_tokens.shape[0]:
        raise ValueError("initial state and tokens must align")
    if min(beam_width, candidates_per_beam, chunk_length, depth) < 1:
        raise ValueError("beam settings must be positive")
    latent = initial_latent
    tokens = initial_tokens
    root_count = len(tokens)
    root_ids = torch.arange(root_count, device=tokens.device)
    log_sum = torch.zeros(len(tokens), device=tokens.device)
    score = torch.zeros_like(log_sum)
    for _ in range(depth):
        candidate_tokens, candidate_log_prob = generate_candidates(
            latent, tokens, candidates_per_beam
        )
        expected = (len(tokens), candidates_per_beam, chunk_length)
        if candidate_tokens.shape != expected:
            raise ValueError(
                f"candidate generator returned {candidate_tokens.shape}, "
                f"expected {expected}"
            )
        flat_chunks = candidate_tokens.flatten(0, 1)
        repeated_latent = latent.repeat_interleave(candidates_per_beam, 0)
        repeated_tokens = tokens.repeat_interleave(candidates_per_beam, 0)
        candidate_roots = root_ids.repeat_interleave(candidates_per_beam)
        leaf = advance(repeated_latent, flat_chunks)
        flat_log = candidate_log_prob.reshape(-1)
        cumulative = log_sum.repeat_interleave(candidates_per_beam) + flat_log
        expanded_tokens = torch.cat([repeated_tokens, flat_chunks], dim=-1)
        leaf_score = score_leaves(leaf, expanded_tokens, cumulative)
        selected_parts = []
        for root in range(root_count):
            members = torch.nonzero(candidate_roots == root).flatten()
            keep = min(beam_width, len(members))
            if keep:
                within = torch.topk(leaf_score[members], keep).indices
                selected_parts.append(members[within])
        selected = torch.cat(selected_parts)
        latent = leaf[selected]
        tokens = expanded_tokens[selected]
        log_sum = cumulative[selected]
        score = leaf_score[selected]
        root_ids = candidate_roots[selected]
    return BeamState(tokens=tokens, latent=latent,
                     log_probability_sum=log_sum, score=score,
                     root_ids=root_ids)


def potential_shaped_reward(
    reward: torch.Tensor,
    current_distance: torch.Tensor,
    next_distance: torch.Tensor,
    *,
    gamma: float,
    weight: float,
    clip: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= gamma <= 1.0 or weight < 0 or clip <= 0:
        raise ValueError("invalid potential-shaping parameters")
    # Φ = -distance, hence γΦ_next - Φ_current = d_current - γd_next.
    increment = weight * (current_distance - gamma * next_distance)
    increment = increment.clamp(-clip, clip)
    return reward + increment, increment
