"""Oracle, value-guided, and worker planning for hierarchical language JEPA."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import torch

from textjepa.objectives.hierarchical_language import (
    EMAShrunkMahalanobis,
    supported_step_cost,
    terminal_set_discrepancy,
    value_teacher_softmin,
)


Tensor = torch.Tensor


@dataclass
class TokenOracleResult:
    costs: Tensor
    weights: Tensor
    next_token_probability: Tensor
    best_index: int


def score_token_candidates(
    predicted_endpoints: Tensor,
    candidate_tokens: Tensor,
    candidate_log_probability: Tensor,
    waypoint: Tensor,
    sentence_encoder: Callable[[Tensor], Tensor] | None = None,
    metric: EMAShrunkMahalanobis | None = None,
    *,
    fine_to_coarse: Callable[[Tensor], Tensor] | None = None,
    prior_weight: float,
    temperature: float,
    vocab_size: int,
) -> TokenOracleResult:
    """Reweight complete autoregressive spans by coarse-waypoint progress."""
    if sentence_encoder is None:
        sentence_encoder = fine_to_coarse
    if sentence_encoder is None or metric is None:
        raise ValueError("sentence encoder and metric are required")
    if temperature <= 0:
        raise ValueError("candidate temperature must be positive")
    if prior_weight < 0:
        raise ValueError("prior weight must be nonnegative")
    if candidate_tokens.ndim != 2 or candidate_tokens.shape[1] < 1:
        raise ValueError("candidate token spans must be nonempty matrices")
    if len(candidate_tokens) != len(predicted_endpoints):
        raise ValueError("one endpoint is required per token span")
    if candidate_log_probability.shape != (len(candidate_tokens),):
        raise ValueError("candidate log probability must be a vector")
    if vocab_size < 1 or bool((candidate_tokens < 0).any()) or bool(
        (candidate_tokens >= vocab_size).any()
    ):
        raise ValueError("candidate token ID lies outside vocabulary")
    coarse = sentence_encoder(predicted_endpoints)
    if coarse.ndim != 2 or waypoint.shape != (coarse.shape[-1],):
        raise ValueError("coarse endpoint and waypoint dimensions disagree")
    costs = metric(coarse, waypoint.expand_as(coarse))
    costs = costs - prior_weight * candidate_log_probability
    weights = torch.softmax(-costs / temperature, 0)
    marginal = torch.zeros(
        vocab_size, device=weights.device, dtype=weights.dtype
    )
    marginal.scatter_add_(0, candidate_tokens[:, 0], weights)
    return TokenOracleResult(
        costs=costs,
        weights=weights,
        next_token_probability=marginal,
        best_index=int(costs.argmin()),
    )


def score_token_space_candidates(
    predicted_endpoints: Tensor,
    candidate_tokens: Tensor,
    candidate_log_probability: Tensor,
    token_waypoint: Tensor,
    metric: EMAShrunkMahalanobis,
    *,
    prior_weight: float,
    temperature: float,
    vocab_size: int,
) -> TokenOracleResult:
    """Flat oracle-token diagnostic; no sentence encoder is involved."""
    return score_token_candidates(
        predicted_endpoints, candidate_tokens, candidate_log_probability,
        token_waypoint, lambda value: value, metric,
        prior_weight=prior_weight, temperature=temperature,
        vocab_size=vocab_size,
    )


@dataclass
class ExactEndpointControl:
    predicted: TokenOracleResult
    exact: TokenOracleResult
    endpoint_cost_gap: Tensor


def exact_endpoint_control(
    predicted_endpoints: Tensor,
    exact_endpoints: Tensor,
    candidate_tokens: Tensor,
    candidate_log_probability: Tensor,
    waypoint: Tensor,
    sentence_encoder: Callable[[Tensor], Tensor] | None,
    metric: EMAShrunkMahalanobis,
    *,
    prior_weight: float,
    temperature: float,
    vocab_size: int,
    fine_to_coarse: Callable[[Tensor], Tensor] | None = None,
) -> ExactEndpointControl:
    if sentence_encoder is None:
        sentence_encoder = fine_to_coarse
    kwargs = dict(
        candidate_tokens=candidate_tokens,
        candidate_log_probability=candidate_log_probability,
        waypoint=waypoint,
        sentence_encoder=sentence_encoder,
        metric=metric,
        prior_weight=prior_weight,
        temperature=temperature,
        vocab_size=vocab_size,
    )
    predicted = score_token_candidates(
        predicted_endpoints=predicted_endpoints, **kwargs
    )
    exact = score_token_candidates(
        predicted_endpoints=exact_endpoints, **kwargs
    )
    return ExactEndpointControl(
        predicted, exact, predicted.costs - exact.costs
    )


def oracle_high_level_cost(
    states: Tensor,
    log_probabilities: Tensor,
    goals: Tensor,
    metric: EMAShrunkMahalanobis,
    *,
    terminal_temperature: float,
    step_cost: float,
    prior_weight: float,
    goal_mask: Tensor | None = None,
) -> Tensor:
    terminal = terminal_set_discrepancy(
        states[:, -1], goals, metric, terminal_temperature, goal_mask
    )
    action_cost = supported_step_cost(
        log_probabilities, step_cost, prior_weight
    ).sum(-1)
    return terminal + action_cost


def oracle_high_level_prefix_cost(
    states: Tensor,
    log_probabilities: Tensor,
    goals: Tensor,
    metric: EMAShrunkMahalanobis,
    *,
    terminal_temperature: float,
    step_cost: float,
    prior_weight: float,
    goal_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Score every 1..K1 prefix and select variable effective lookahead."""
    if states.ndim != 3 or log_probabilities.shape != states.shape[:2]:
        raise ValueError("high-level trajectories must be [N,K,*]")
    cumulative = supported_step_cost(
        log_probabilities, step_cost, prior_weight
    ).cumsum(-1)
    terminal = torch.stack([
        terminal_set_discrepancy(
            states[:, step], goals, metric, terminal_temperature, goal_mask
        )
        for step in range(states.shape[1])
    ], -1)
    prefix_cost = cumulative + terminal
    return prefix_cost.min(-1)


@torch.no_grad()
def build_terminal_state_set(
    model,
    hidden_states: Tensor,
    solution_end: Tensor,
    trace_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Encode complete verified traces immediately before chat EOS."""
    if hidden_states.ndim != 4 or solution_end.shape != hidden_states.shape[:2]:
        raise ValueError("terminal traces must be [problems,traces,tokens,D]")
    if trace_mask.shape != solution_end.shape or trace_mask.dtype != torch.bool:
        raise ValueError("trace mask must align with terminal traces")
    if bool((~trace_mask.any(-1)).any()):
        raise ValueError("every problem needs a verified successful trace")
    rows, traces = solution_end.shape
    flat_hidden = hidden_states.reshape(
        rows * traces, hidden_states.shape[2], hidden_states.shape[3]
    )
    flat_end = solution_end.reshape(-1)
    if bool((flat_end[trace_mask.reshape(-1)] < 1).any()) or bool(
        (flat_end[trace_mask.reshape(-1)] > hidden_states.shape[2]).any()
    ):
        raise ValueError("terminal prefix length lies outside hidden states")
    index = torch.arange(rows * traces, device=hidden_states.device)
    endpoint = flat_hidden[index, flat_end.clamp_min(1) - 1]
    goals = model.encode_sentence(
        endpoint, target=True
    ).reshape(rows, traces, -1)
    return goals, trace_mask


def construct_value_teacher(
    initial_state: Tensor,
    initial_context: Tensor,
    first_actions: Tensor,
    continuation_actions: Tensor,
    continuation_mask: Tensor,
    goals: Tensor,
    goal_mask: Tensor,
    advance: Callable[
        [Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]
    ],
    metric: EMAShrunkMahalanobis,
    *,
    terminal_temperature: float,
    teacher_temperature: float,
    step_cost: float,
    prior_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Offline multi-sample/multi-depth oracle teacher for first actions.

    ``advance`` returns successor state, successor pre-action context, and the
    causal prior log probability of the supplied action.
    """
    if first_actions.ndim != 3 or continuation_actions.ndim != 5:
        raise ValueError("value teacher actions have invalid ranks")
    batch, first_count, samples, depth, action_dim = continuation_actions.shape
    if first_actions.shape != (batch, first_count, action_dim):
        raise ValueError("first and continuation actions do not align")
    if continuation_mask.shape != continuation_actions.shape[:-1]:
        raise ValueError("continuation mask does not align")
    if bool((~continuation_mask[..., :-1] & continuation_mask[..., 1:]).any()):
        raise ValueError("continuation depth masks must be right-contiguous")
    branches = batch * first_count * samples
    state = initial_state[:, None, None].expand(
        batch, first_count, samples, initial_state.shape[-1]
    ).reshape(branches, -1)
    context = initial_context[:, None, None].expand(
        batch, first_count, samples, initial_context.shape[-1]
    ).reshape(branches, -1)
    first = first_actions[:, :, None].expand(
        batch, first_count, samples, action_dim
    ).reshape(branches, action_dim)
    state, context, first_logp = advance(state, context, first)
    accumulated = supported_step_cost(
        first_logp, step_cost, prior_weight
    )
    # Slot zero is the one-action plan consisting only of the sampled first
    # action. Remaining slots add continuation actions. Thus prefix indices
    # denote total plan lengths 1..(depth + 1), not continuation depth.
    costs = state.new_full((branches, depth + 1), torch.inf)
    flat_actions = continuation_actions.reshape(
        branches, depth, action_dim
    )
    flat_mask = continuation_mask.reshape(branches, depth)
    flat_goals = goals[:, None, None].expand(
        batch, first_count, samples, *goals.shape[1:]
    ).reshape(branches, *goals.shape[1:])
    flat_goal_mask = goal_mask[:, None, None].expand(
        batch, first_count, samples, goal_mask.shape[-1]
    ).reshape(branches, goal_mask.shape[-1])
    costs[:, 0] = accumulated + terminal_set_discrepancy(
        state, flat_goals, metric, terminal_temperature, flat_goal_mask
    )
    for step in range(depth):
        state, context, logp = advance(
            state, context, flat_actions[:, step]
        )
        accumulated = accumulated + supported_step_cost(
            logp, step_cost, prior_weight
        )
        terminal = terminal_set_discrepancy(
            state, flat_goals, metric, terminal_temperature,
            flat_goal_mask,
        )
        costs[:, step + 1] = accumulated + terminal
    costs = costs.reshape(batch, first_count, samples, depth + 1)
    total_prefix_mask = torch.cat([
        torch.ones(
            *continuation_mask.shape[:-1], 1,
            dtype=torch.bool, device=continuation_mask.device,
        ),
        continuation_mask,
    ], -1)
    target = value_teacher_softmin(
        costs, total_prefix_mask, teacher_temperature
    )
    successor_state = state.new_empty(
        batch, first_count, initial_state.shape[-1]
    )
    successor_context = context.new_empty(
        batch, first_count, initial_context.shape[-1]
    )
    # Recompute first successors once per action rather than selecting a
    # continuation-dependent terminal context.
    state0 = initial_state[:, None].expand(
        batch, first_count, initial_state.shape[-1]
    ).reshape(batch * first_count, -1)
    context0 = initial_context[:, None].expand(
        batch, first_count, initial_context.shape[-1]
    ).reshape(batch * first_count, -1)
    first0 = first_actions.reshape(batch * first_count, action_dim)
    state1, context1, _ = advance(state0, context0, first0)
    successor_state.copy_(state1.reshape_as(successor_state))
    successor_context.copy_(context1.reshape_as(successor_context))
    return target, successor_state, successor_context


def construct_value_teacher_from_rollouts(
    rollout_states: Tensor,
    rollout_log_probabilities: Tensor,
    rollout_mask: Tensor,
    goals: Tensor,
    goal_mask: Tensor,
    metric: EMAShrunkMahalanobis,
    *,
    terminal_temperature: float,
    teacher_temperature: float,
    step_cost: float,
    prior_weight: float,
) -> Tensor:
    """Build first-action targets from grounded total-prefix rollouts.

    Shapes are ``[batch, first_actions, samples, total_prefixes, ...]``.
    Prefix zero is always the one-action plan.
    """
    if rollout_states.ndim != 5 or (
        rollout_log_probabilities.shape != rollout_states.shape[:-1]
    ) or rollout_mask.shape != rollout_states.shape[:-1]:
        raise ValueError("value-teacher rollout tensors do not align")
    if rollout_mask.dtype != torch.bool or bool(
        (~rollout_mask[..., :-1] & rollout_mask[..., 1:]).any()
    ):
        raise ValueError("value-teacher prefix mask must be right-contiguous")
    if bool((~rollout_mask[..., 0]).any()):
        raise ValueError("every rollout must include its one-action prefix")
    batch, first, samples, prefixes, _ = rollout_states.shape
    flat_states = rollout_states.reshape(-1, rollout_states.shape[-1])
    flat_goals = goals[:, None, None, None].expand(
        batch, first, samples, prefixes, *goals.shape[1:]
    ).reshape(-1, *goals.shape[1:])
    flat_goal_mask = goal_mask[:, None, None, None].expand(
        batch, first, samples, prefixes, goal_mask.shape[-1]
    ).reshape(-1, goal_mask.shape[-1])
    terminal = terminal_set_discrepancy(
        flat_states, flat_goals, metric, terminal_temperature,
        flat_goal_mask,
    ).reshape(batch, first, samples, prefixes)
    cumulative = supported_step_cost(
        rollout_log_probabilities, step_cost, prior_weight
    ).cumsum(-1)
    return value_teacher_softmin(
        terminal + cumulative, rollout_mask, teacher_temperature
    )


def value_guided_high_level_cost(
    contexts: Tensor,
    task: Tensor,
    log_probabilities: Tensor,
    value: Callable[[Tensor, Tensor], Tensor],
    *,
    step_cost: float,
    prior_weight: float,
) -> Tensor:
    """Explicit K1-step cost plus unbudgeted continuation value."""
    action_cost = supported_step_cost(
        log_probabilities, step_cost, prior_weight
    ).sum(-1)
    final_context = contexts[:, -1] if contexts.ndim == 3 else contexts
    if final_context.ndim != 2:
        raise ValueError("contexts must be final [N,D] or trajectory [N,K,D]")
    return action_cost + value(final_context, task)


def value_guided_high_level_prefix_cost(
    states: Tensor,
    contexts: Tensor,
    task: Tensor,
    log_probabilities: Tensor,
    value: Callable[[Tensor, Tensor, Tensor], Tensor],
    *,
    step_cost: float,
    prior_weight: float,
) -> tuple[Tensor, Tensor]:
    """Variable 1..K1 lookahead with V on each predicted successor."""
    if states.ndim != 3 or contexts.ndim != 3:
        raise ValueError("state and context trajectories must be [N,K,D]")
    if states.shape[:2] != contexts.shape[:2] or (
        log_probabilities.shape != states.shape[:2]
    ):
        raise ValueError("high-level value trajectories do not align")
    cumulative = supported_step_cost(
        log_probabilities, step_cost, prior_weight
    ).cumsum(-1)
    task_steps = task[:, None].expand(
        states.shape[0], states.shape[1], task.shape[-1]
    )
    prefix = cumulative + value(states, contexts, task_steps)
    return prefix.min(-1)


@dataclass
class PopulationSearchResult:
    tokens: Tensor
    cost: Tensor
    diagnostics: list[dict[str, float]]


def autoregressive_population_search(
    sample: Callable[[Tensor | None, int], tuple[Tensor, Tensor]],
    score: Callable[[Tensor, Tensor], Tensor],
    *,
    population: int,
    iterations: int,
    elite_fraction: float,
    preserve_prefix: int,
) -> PopulationSearchResult:
    """Elite-prefix regeneration without factorizing future token positions.

    ``sample(prefixes, population)`` must sample complete spans
    autoregressively from the frozen LM and return spans and their log
    probabilities.  On later iterations, prefixes are selected from entire
    elite spans and suffixes are regenerated by the same callback.
    """
    if population < 1 or iterations < 1 or not 0 < elite_fraction <= 1:
        raise ValueError("invalid population-search configuration")
    prefixes = None
    best_tokens, best_cost = None, None
    diagnostics = []
    for iteration in range(iterations):
        tokens, log_probability = sample(prefixes, population)
        costs = score(tokens, log_probability)
        if costs.shape != (population,):
            raise ValueError("score must return one cost per candidate")
        elite_count = max(1, round(population * elite_fraction))
        elite_ids = costs.topk(elite_count, largest=False).indices
        elite = tokens[elite_ids]
        width = min(max(0, preserve_prefix), tokens.shape[1])
        prefixes = elite[:, :width]
        index = int(costs.argmin())
        if best_cost is None or bool(costs[index] < best_cost):
            best_tokens, best_cost = tokens[index].clone(), costs[index].clone()
        diagnostics.append({
            "iteration": float(iteration),
            "best_cost": float(costs.min()),
            "median_cost": float(costs.median()),
            "elite_unique": float(torch.unique(elite, dim=0).shape[0]),
        })
    assert best_tokens is not None and best_cost is not None
    return PopulationSearchResult(best_tokens, best_cost, diagnostics)


@dataclass
class PriorCEMResult:
    noise: Tensor
    actions: Tensor
    states: Tensor
    cost: float
    diagnostics: list[dict[str, float]]
    grounded_replay: list[dict[str, Tensor]]


def prior_coordinate_cem(
    advance: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    prior: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    objective: Callable[[Tensor, Tensor, Tensor], Tensor],
    initial_state: Tensor,
    initial_context: Tensor,
    task: Tensor,
    *,
    horizon: int,
    action_dim: int,
    population: int = 256,
    iterations: int = 4,
    elite_fraction: float = 0.1,
    smoothing: float = 0.1,
    covariance_floor: float = 0.05,
    trust_region: float = 3.0,
    ground: Callable[[Tensor, Tensor], dict[str, Tensor]] | None = None,
    ground_topn: int = 0,
    ground_random: int = 0,
    select_grounded: bool = False,
    prior_flops_per_candidate_step: float = 0.0,
    advance_flops_per_candidate_step: float = 0.0,
    objective_flops_per_candidate: float = 0.0,
    grounding_flops_per_candidate: float = 0.0,
    generator: torch.Generator | None = None,
) -> PriorCEMResult:
    """Continuous CEM in standard-normal prior coordinates."""
    if horizon < 1 or action_dim < 1 or population < 1 or iterations < 1:
        raise ValueError("invalid CEM shape")
    if not 0 < elite_fraction <= 1 or not 0 <= smoothing < 1:
        raise ValueError("invalid CEM update parameter")
    if covariance_floor <= 0:
        raise ValueError("covariance floor must be positive")
    if trust_region < 0:
        raise ValueError("trust region must be nonnegative")
    if ground_topn < 0 or ground_random < 0:
        raise ValueError("grounding sample counts must be nonnegative")
    if select_grounded and (
        ground is None or ground_topn + ground_random < 1
    ):
        raise ValueError(
            "grounded selection requires a ground callback and candidates"
        )
    flop_rates = (
        prior_flops_per_candidate_step,
        advance_flops_per_candidate_step,
        objective_flops_per_candidate,
        grounding_flops_per_candidate,
    )
    if any(value < 0 for value in flop_rates):
        raise ValueError("planning FLOP rates must be nonnegative")
    mean = initial_state.new_zeros(horizon, action_dim)
    std = initial_state.new_ones(horizon, action_dim)
    elite_count = max(1, round(population * elite_fraction))
    best = None
    diagnostics = []
    grounded_replay = []
    for iteration in range(iterations):
        iteration_started = perf_counter()
        prior_seconds = 0.0
        advance_seconds = 0.0
        objective_seconds = 0.0
        grounding_seconds = 0.0
        noise = mean + std * torch.randn(
            population, horizon, action_dim,
            device=mean.device, dtype=mean.dtype, generator=generator,
        )
        norm = noise.square().mean((-1, -2)).sqrt()
        state = initial_state.reshape(1, -1).expand(population, -1)
        context = initial_context.reshape(1, -1).expand(population, -1)
        actions, states, contexts, log_probabilities = [], [], [], []
        for step in range(horizon):
            task_batch = task.reshape(1, -1).expand(population, -1)
            started = perf_counter()
            prior_mean, prior_logvar = prior(state, context, task_batch)
            prior_seconds += perf_counter() - started
            action = prior_mean + (0.5 * prior_logvar).exp() * noise[:, step]
            log_probability = -0.5 * (
                noise[:, step].square()
                + prior_logvar
                + torch.log(torch.tensor(
                    2 * torch.pi, device=mean.device, dtype=mean.dtype
                ))
            ).sum(-1)
            started = perf_counter()
            state, context = advance(state, context, action)
            advance_seconds += perf_counter() - started
            actions.append(action)
            states.append(state)
            contexts.append(context)
            log_probabilities.append(log_probability)
        actions_t = torch.stack(actions, 1)
        states_t = torch.stack(states, 1)
        contexts_t = torch.stack(contexts, 1)
        logp_t = torch.stack(log_probabilities, 1)
        started = perf_counter()
        cost = objective(states_t, contexts_t, logp_t)
        objective_seconds += perf_counter() - started
        cost = cost + (norm - trust_region).clamp_min(0).square()
        selection_cost = cost
        grounded: dict[str, Tensor] = {}
        grounded_ids = None
        if ground is not None and (ground_topn or ground_random):
            top = cost.topk(min(ground_topn, population), largest=False).indices
            remaining = torch.ones(
                population, dtype=torch.bool, device=cost.device
            )
            remaining[top] = False
            pool = remaining.nonzero().flatten()
            random_count = min(ground_random, len(pool))
            random_ids = (
                pool[torch.randperm(
                    len(pool), device=pool.device, generator=generator
                )[:random_count]]
                if random_count else top[:0]
            )
            grounded_ids = torch.unique(torch.cat([top, random_ids]))
            started = perf_counter()
            grounded = ground(
                actions_t[grounded_ids], states_t[grounded_ids]
            )
            grounding_seconds += perf_counter() - started
            if not grounded or any(
                value.shape != (len(grounded_ids),)
                for value in grounded.values()
            ):
                raise ValueError(
                    "ground callback must return aligned scalar cost vectors"
                )
            grounded_replay.append({
                "iteration": torch.full(
                    (len(grounded_ids),), iteration,
                    dtype=torch.long, device=cost.device,
                ),
                "candidate_ids": grounded_ids.detach().clone(),
                "actions": actions_t[grounded_ids].detach().clone(),
                "predicted_states": states_t[grounded_ids].detach().clone(),
                "predicted_cost": cost[grounded_ids].detach().clone(),
                **{
                    name: value.detach().clone()
                    for name, value in grounded.items()
                },
            })
            if select_grounded:
                selection_cost = grounded.get(
                    "achieved_cost", grounded.get("exact_cost")
                )
                if selection_cost is None:
                    raise ValueError(
                        "grounded selection needs achieved_cost or exact_cost"
                    )
                grounded_selection = selection_cost
                selection_cost = torch.full_like(cost, torch.inf)
                selection_cost[grounded_ids] = grounded_selection
        if select_grounded and grounded_ids is not None:
            grounded_elite_count = min(elite_count, len(grounded_ids))
            local_elites = grounded_selection.topk(
                grounded_elite_count, largest=False
            ).indices
            elite_ids = grounded_ids[local_elites]
        else:
            elite_ids = cost.topk(elite_count, largest=False).indices
        elite = noise[elite_ids]
        new_mean = elite.mean(0)
        new_std = elite.std(0, unbiased=False).clamp_min(covariance_floor)
        mean = smoothing * mean + (1 - smoothing) * new_mean
        std = smoothing * std + (1 - smoothing) * new_std
        index = int(cost.argmin())
        if select_grounded and grounded_ids is not None:
            index = int(selection_cost.argmin())
        if best is None or float(selection_cost[index]) < best[0]:
            best = (
                float(selection_cost[index]), noise[index].clone(),
                actions_t[index].clone(), states_t[index].clone(),
            )
        row = {
            "iteration": float(iteration),
            "best_predicted_cost": float(cost.min()),
            "median_predicted_cost": float(cost.median()),
            "elite_diversity": float(elite.std(0, unbiased=False).mean()),
            "prior_noise_norm": float(norm[elite_ids].mean()),
            "cem_std": float(std.mean()),
            "unique_plans": float(torch.unique(actions_t, dim=0).shape[0]),
            "prior_seconds": prior_seconds,
            "advance_seconds": advance_seconds,
            "objective_seconds": objective_seconds,
            "grounding_seconds": grounding_seconds,
            "iteration_seconds": perf_counter() - iteration_started,
        }
        component_flops = {
            "prior": prior_flops_per_candidate_step * population * horizon,
            "advance": advance_flops_per_candidate_step * population * horizon,
            "objective": objective_flops_per_candidate * population,
            "grounding": grounding_flops_per_candidate * (
                len(grounded_ids) if grounded_ids is not None else 0
            ),
        }
        total_iteration_flops = sum(component_flops.values())
        row["estimated_flops"] = total_iteration_flops
        for name, value in component_flops.items():
            row[f"{name}_estimated_flops"] = value
            row[f"{name}_flop_fraction"] = (
                value / total_iteration_flops
                if total_iteration_flops else 0.0
            )
        if grounded:
            for name, value in grounded.items():
                row[f"best_{name}"] = float(value.min())
                selected_local = (
                    int((grounded_ids == index).nonzero()[0])
                    if bool((grounded_ids == index).any())
                    else int(cost[grounded_ids].argmin())
                )
                row[f"selected_{name}"] = float(value[selected_local])
        diagnostics.append(row)
    assert best is not None
    return PriorCEMResult(
        noise=best[1], actions=best[2], states=best[3],
        cost=best[0], diagnostics=diagnostics,
        grounded_replay=grounded_replay,
    )


def optimizer_curse_curve(
    predicted_cost: Tensor,
    exact_cost: Tensor,
    population_sizes: Sequence[int],
) -> list[dict[str, float]]:
    """Compare predicted selection quality with exact-grounded quality."""
    if predicted_cost.shape != exact_cost.shape or predicted_cost.ndim != 1:
        raise ValueError("optimizer-curse inputs must be aligned vectors")
    curve = []
    for size in population_sizes:
        if size < 1 or size > len(predicted_cost):
            raise ValueError("population size lies outside supplied candidates")
        index = int(predicted_cost[:size].argmin())
        curve.append({
            "population": float(size),
            "best_predicted": float(predicted_cost[index]),
            "selected_exact": float(exact_cost[index]),
            "best_exact": float(exact_cost[:size].min()),
        })
    return curve


@dataclass
class MPCStep:
    planned_text: Tensor
    executed_text: Tensor
    exact_token_state: Tensor
    exact_sentence_state: Tensor
    manager_seconds: float = 0.0
    worker_seconds: float = 0.0
    exact_reencode_seconds: float = 0.0


def receding_horizon_step(
    planned_text: Tensor,
    execute: Callable[[Tensor], Tensor],
    exact_encode: Callable[[Tensor], tuple[Tensor, Tensor]],
    *,
    n_exec: int,
) -> MPCStep:
    """Execute a prefix, then discard imagined states and re-encode exactly."""
    if n_exec < 1:
        raise ValueError("n_exec must be positive")
    executed = execute(planned_text[:n_exec])
    token_state, sentence_state = exact_encode(executed)
    return MPCStep(planned_text, executed, token_state, sentence_state)


@dataclass
class HierarchicalMPCResult:
    executed_tokens: Tensor
    steps: list[MPCStep]
    manager_warm_start: object | None
    worker_warm_start: object | None


def hierarchical_mpc(
    initial_prefix: Tensor,
    manager_plan: Callable[[Tensor, object | None], tuple[Tensor, object]],
    worker_plan: Callable[
        [Tensor, Tensor, object | None], tuple[Tensor, object]
    ],
    append_and_encode: Callable[
        [Tensor, Tensor], tuple[Tensor, Tensor, Tensor, object, object]
    ],
    *,
    n_exec: int,
    replans: int,
) -> HierarchicalMPCResult:
    """Closed-loop manager -> waypoint -> worker -> exact-cache rebuild."""
    if n_exec < 1 or replans < 1:
        raise ValueError("n_exec and replans must be positive")
    prefix = initial_prefix
    manager_warm = worker_warm = None
    steps = []
    for _ in range(replans):
        started = perf_counter()
        waypoint, manager_warm = manager_plan(prefix, manager_warm)
        manager_seconds = perf_counter() - started
        started = perf_counter()
        planned, worker_warm = worker_plan(prefix, waypoint, worker_warm)
        worker_seconds = perf_counter() - started
        executed = planned[:n_exec]
        started = perf_counter()
        (
            prefix, token_state, sentence_state,
            manager_cache, worker_cache,
        ) = append_and_encode(
            prefix, executed
        )
        exact_seconds = perf_counter() - started
        manager_warm, worker_warm = manager_cache, worker_cache
        steps.append(MPCStep(
            planned_text=planned,
            executed_text=executed,
            exact_token_state=token_state,
            exact_sentence_state=sentence_state,
            manager_seconds=manager_seconds,
            worker_seconds=worker_seconds,
            exact_reencode_seconds=exact_seconds,
        ))
    return HierarchicalMPCResult(
        prefix, steps, manager_warm, worker_warm
    )
