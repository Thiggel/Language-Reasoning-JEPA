"""Dependency-preserving discrete search for token-JEPA waypoint control.

The optimizers in this module never assume that future token positions are
independent.  Beam search asks an autoregressive proposal for supported next
tokens at every live prefix.  Elite-prefix CEM resamples complete suffixes
conditioned on intact elite prefixes.  Markov CEM is an explicit, deliberately
weaker first-order ablation whose transition support is learned from complete
autoregressive proposal trajectories.

All semantic scores are supplied by the caller.  In the hierarchical language
runtime that score is the sentence-space discrepancy after rolling the token
JEPA forward and applying E0_to_1.  Proposal log probability is recorded but
is not part of the score unless an experiment explicitly requests it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TokenTrajectory:
    tokens: Tensor
    terminal: bool
    log_probability: float


@dataclass(frozen=True)
class TokenSearchResult:
    trajectory: TokenTrajectory
    predicted_endpoint: Tensor
    cost: float
    diagnostics: list[dict[str, float]]
    proposed_tokens: int
    transition_evaluations: int


def _validate_trajectory(trajectory: TokenTrajectory) -> None:
    if trajectory.tokens.ndim != 1 or len(trajectory.tokens) < 1:
        raise ValueError("token trajectories must be nonempty vectors")
    if trajectory.tokens.dtype != torch.long:
        raise ValueError("token trajectories must contain integer token IDs")
    if not torch.isfinite(torch.tensor(trajectory.log_probability)):
        raise ValueError("trajectory log probability must be finite")


def _score(
    semantic_cost: Tensor,
    log_probability: Tensor,
    *,
    objective: str,
    prior_weight: float,
) -> Tensor:
    if semantic_cost.ndim != 1 or log_probability.shape != semantic_cost.shape:
        raise ValueError("search costs must be aligned vectors")
    if objective not in {"jepa", "combined", "lm"}:
        raise ValueError("unknown token-search objective")
    if prior_weight < 0:
        raise ValueError("prior weight must be nonnegative")
    if objective == "jepa":
        return semantic_cost
    if objective == "lm":
        return -log_probability
    return semantic_cost - prior_weight * log_probability


def _deduplicate(
    trajectories: Iterable[TokenTrajectory],
) -> list[TokenTrajectory]:
    best: dict[tuple[int, ...], TokenTrajectory] = {}
    for trajectory in trajectories:
        _validate_trajectory(trajectory)
        key = tuple(int(token) for token in trajectory.tokens.tolist())
        previous = best.get(key)
        if previous is None or (
            trajectory.log_probability > previous.log_probability
        ):
            best[key] = trajectory
    return list(best.values())


def jepa_beam_search(
    propose: Callable[[Sequence[Tensor], int], tuple[Tensor, Tensor]],
    rollout: Callable[[Sequence[Tensor]], Tensor],
    semantic_cost: Callable[[Tensor], Tensor],
    is_complete: Callable[[Tensor], tuple[bool, bool, Tensor]],
    *,
    horizon: int,
    beam_width: int,
    branch_factor: int,
    objective: str = "jepa",
    prior_weight: float = 0.0,
) -> TokenSearchResult:
    """Autoregressive JEPA beam search over supported next-token expansions.

    ``propose`` returns top supported token IDs and their conditional log
    probabilities for every supplied complete prefix. ``is_complete`` returns
    ``(complete, terminal, stored_tokens)``; EOS can therefore terminate a
    trajectory without becoming a JEPA action, while a newline delimiter can
    remain in the stored action.
    """

    if min(horizon, beam_width, branch_factor) < 1:
        raise ValueError("beam-search sizes must be positive")
    active: list[TokenTrajectory] = [TokenTrajectory(
        torch.empty(0, dtype=torch.long), False, 0.0
    )]
    completed: list[TokenTrajectory] = []
    diagnostics: list[dict[str, float]] = []
    proposed_tokens = transition_evaluations = 0
    endpoint_by_key: dict[tuple[int, ...], Tensor] = {}
    cost_by_key: dict[tuple[int, ...], float] = {}

    for depth in range(1, horizon + 1):
        prefixes = [trajectory.tokens for trajectory in active]
        token_ids, conditional_logp = propose(prefixes, branch_factor)
        if token_ids.shape != conditional_logp.shape or token_ids.shape != (
            len(active), branch_factor
        ):
            raise ValueError("proposal must return [prefixes, branch] tensors")
        if token_ids.dtype != torch.long or not bool(
            torch.isfinite(conditional_logp).all()
        ):
            raise ValueError("proposal tokens/log probabilities are invalid")
        proposed_tokens += token_ids.numel()
        children: list[TokenTrajectory] = []
        for parent_index, parent in enumerate(active):
            for branch in range(branch_factor):
                raw = torch.cat([
                    parent.tokens,
                    token_ids[parent_index, branch:branch + 1].cpu(),
                ])
                done, terminal, stored = is_complete(raw)
                if len(stored) == 0:
                    continue
                child = TokenTrajectory(
                    stored.to(dtype=torch.long), bool(terminal),
                    parent.log_probability
                    + float(conditional_logp[parent_index, branch]),
                )
                if done:
                    completed.append(child)
                else:
                    children.append(child)
        children = _deduplicate(children)
        completed = _deduplicate(completed)
        pool = children + completed
        if not pool:
            raise RuntimeError("beam search exhausted all supported prefixes")
        unseen = [
            trajectory for trajectory in pool
            if tuple(trajectory.tokens.tolist()) not in endpoint_by_key
        ]
        if unseen:
            new_endpoints = rollout([
                trajectory.tokens for trajectory in unseen
            ])
            if new_endpoints.ndim != 2 or len(new_endpoints) != len(unseen):
                raise ValueError(
                    "rollout must return one endpoint per trajectory"
                )
            transition_evaluations += sum(
                len(item.tokens) for item in unseen
            )
            for item, endpoint in zip(unseen, new_endpoints):
                endpoint_by_key[tuple(item.tokens.tolist())] = (
                    endpoint.detach().clone()
                )
        endpoints = torch.stack([
            endpoint_by_key[tuple(item.tokens.tolist())] for item in pool
        ])
        semantic = semantic_cost(endpoints)
        logp = semantic.new_tensor([
            trajectory.log_probability for trajectory in pool
        ])
        costs = _score(
            semantic, logp, objective=objective, prior_weight=prior_weight
        )
        for item, endpoint, cost in zip(pool, endpoints, costs):
            key = tuple(item.tokens.tolist())
            endpoint_by_key[key] = endpoint.detach().clone()
            cost_by_key[key] = float(cost)
        if children:
            child_cost = torch.tensor([
                cost_by_key[tuple(item.tokens.tolist())] for item in children
            ])
            keep = child_cost.topk(
                min(beam_width, len(children)), largest=False
            ).indices.tolist()
            active = [children[index] for index in keep]
        else:
            active = []
        diagnostics.append({
            "iteration": float(depth - 1),
            "depth": float(depth),
            "live_prefixes": float(len(active)),
            "complete_trajectories": float(len(completed)),
            "best_complete_cost": (
                min(cost_by_key[tuple(item.tokens.tolist())]
                    for item in completed)
                if completed else float("nan")
            ),
        })
        if not active:
            break
    if not completed:
        raise RuntimeError(
            f"beam search found no complete reasoning step within K0={horizon}"
        )
    winner = min(
        completed, key=lambda item: cost_by_key[tuple(item.tokens.tolist())]
    )
    key = tuple(winner.tokens.tolist())
    return TokenSearchResult(
        winner, endpoint_by_key[key], cost_by_key[key], diagnostics,
        proposed_tokens, transition_evaluations,
    )


def elite_prefix_cem(
    sample: Callable[
        [Sequence[Tensor] | None, int, int], Sequence[TokenTrajectory]
    ],
    rollout: Callable[[Sequence[Tensor]], Tensor],
    semantic_cost: Callable[[Tensor], Tensor],
    *,
    population: int,
    iterations: int,
    elite_fraction: float,
    preserve_prefix: int,
    objective: str = "jepa",
    prior_weight: float = 0.0,
) -> TokenSearchResult:
    """Non-parametric autoregressive CEM with intact elite prefixes."""

    if min(population, iterations) < 1 or not 0 < elite_fraction <= 1:
        raise ValueError("CEM population/iterations/elites are invalid")
    if preserve_prefix < 1:
        raise ValueError("elite-prefix CEM must preserve at least one token")
    elite_prefixes: list[Tensor] | None = None
    best: tuple[float, TokenTrajectory, Tensor] | None = None
    diagnostics: list[dict[str, float]] = []
    proposed_tokens = transition_evaluations = 0
    for iteration in range(iterations):
        trajectories = _deduplicate(sample(
            elite_prefixes, population, iteration
        ))
        if not trajectories:
            raise RuntimeError("autoregressive CEM produced no trajectories")
        endpoints = rollout([item.tokens for item in trajectories])
        semantic = semantic_cost(endpoints)
        logp = semantic.new_tensor([
            item.log_probability for item in trajectories
        ])
        costs = _score(
            semantic, logp, objective=objective, prior_weight=prior_weight
        )
        proposed_tokens += sum(len(item.tokens) for item in trajectories)
        transition_evaluations += sum(
            len(item.tokens) for item in trajectories
        )
        elite_count = max(1, round(population * elite_fraction))
        elite_ids = costs.topk(
            min(elite_count, len(trajectories)), largest=False
        ).indices.tolist()
        elite_prefixes = []
        for index in elite_ids:
            tokens = trajectories[index].tokens
            width = min(preserve_prefix, max(1, len(tokens) - 1))
            elite_prefixes.append(tokens[:width].clone())
        index = int(costs.argmin())
        value = float(costs[index])
        if best is None or value < best[0]:
            best = (
                value, trajectories[index], endpoints[index].detach().clone()
            )
        diagnostics.append({
            "iteration": float(iteration),
            "population": float(len(trajectories)),
            "elite_unique": float(len({
                tuple(prefix.tolist()) for prefix in elite_prefixes
            })),
            "best_cost": float(costs.min()),
            "median_cost": float(costs.median()),
        })
    assert best is not None
    return TokenSearchResult(
        best[1], best[2], best[0], diagnostics,
        proposed_tokens, transition_evaluations,
    )


_START = -1
_STOP_NEWLINE = -2
_STOP_EOS = -3


def _with_stop(trajectory: TokenTrajectory) -> list[int]:
    return trajectory.tokens.tolist() + [
        _STOP_EOS if trajectory.terminal else _STOP_NEWLINE
    ]


def _transition_support(
    trajectories: Sequence[TokenTrajectory],
) -> dict[int, list[int]]:
    support: dict[int, set[int]] = {}
    for trajectory in trajectories:
        previous = _START
        for token in _with_stop(trajectory):
            support.setdefault(previous, set()).add(int(token))
            previous = int(token)
    return {key: sorted(value) for key, value in support.items()}


def first_order_markov_cem(
    initial: Sequence[TokenTrajectory],
    rollout: Callable[[Sequence[Tensor]], Tensor],
    semantic_cost: Callable[[Tensor], Tensor],
    *,
    horizon: int,
    population: int,
    iterations: int,
    elite_fraction: float,
    smoothing: float = 0.1,
    objective: str = "jepa",
    prior_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> TokenSearchResult:
    """Sparse first-order Markov CEM fitted to supported proposal bigrams."""

    if min(horizon, population, iterations) < 1:
        raise ValueError("Markov CEM sizes must be positive")
    if not 0 < elite_fraction <= 1 or not 0 <= smoothing < 1:
        raise ValueError("Markov CEM update parameters are invalid")
    initial = _deduplicate(initial)
    if not initial:
        raise ValueError("Markov CEM needs supported initial trajectories")
    support = _transition_support(initial)
    probabilities = {
        previous: torch.full((len(tokens),), 1.0 / len(tokens))
        for previous, tokens in support.items()
    }
    best: tuple[float, TokenTrajectory, Tensor] | None = None
    diagnostics: list[dict[str, float]] = []
    proposed_tokens = transition_evaluations = 0
    for iteration in range(iterations):
        sampled: list[TokenTrajectory] = []
        attempts = 0
        while len(sampled) < population and attempts < 20 * population:
            attempts += 1
            values: list[int] = []
            previous = _START
            logp = 0.0
            terminal = False
            complete = False
            for _ in range(horizon + 1):
                options = support.get(previous)
                if not options:
                    break
                probs = probabilities[previous]
                if len(values) >= horizon:
                    allowed = torch.tensor([
                        token in {_STOP_NEWLINE, _STOP_EOS}
                        for token in options
                    ])
                    if not bool(allowed.any()):
                        break
                    probs = probs * allowed
                    probs = probs / probs.sum()
                selected = int(torch.multinomial(
                    probs, 1, generator=generator
                ))
                token = options[selected]
                logp += float(probs[selected].clamp_min(1e-12).log())
                if token in {_STOP_NEWLINE, _STOP_EOS}:
                    complete = bool(values)
                    terminal = token == _STOP_EOS
                    break
                values.append(token)
                previous = token
                if len(values) >= horizon:
                    # A delimiter token can itself be the final stored token;
                    # completion still requires a learned stop transition.
                    continue
            if complete:
                sampled.append(TokenTrajectory(
                    torch.tensor(values, dtype=torch.long), terminal, logp
                ))
        trajectories = _deduplicate(sampled)
        if not trajectories:
            raise RuntimeError("Markov CEM produced no complete trajectory")
        endpoints = rollout([item.tokens for item in trajectories])
        semantic = semantic_cost(endpoints)
        logp = semantic.new_tensor([
            item.log_probability for item in trajectories
        ])
        costs = _score(
            semantic, logp, objective=objective, prior_weight=prior_weight
        )
        proposed_tokens += sum(len(item.tokens) for item in trajectories)
        transition_evaluations += sum(
            len(item.tokens) for item in trajectories
        )
        elite_count = max(1, round(population * elite_fraction))
        elite_ids = costs.topk(
            min(elite_count, len(trajectories)), largest=False
        ).indices.tolist()
        counts = {
            previous: torch.full((len(tokens),), 1e-3)
            for previous, tokens in support.items()
        }
        for index in elite_ids:
            previous = _START
            for token in _with_stop(trajectories[index]):
                option_index = support[previous].index(int(token))
                counts[previous][option_index] += 1
                previous = int(token)
        for previous in probabilities:
            updated = counts[previous] / counts[previous].sum()
            probabilities[previous] = (
                smoothing * probabilities[previous]
                + (1 - smoothing) * updated
            )
        index = int(costs.argmin())
        value = float(costs[index])
        if best is None or value < best[0]:
            best = (
                value, trajectories[index], endpoints[index].detach().clone()
            )
        entropy = torch.stack([
            -(prob.clamp_min(1e-12).log() * prob).sum()
            for prob in probabilities.values()
        ]).mean()
        diagnostics.append({
            "iteration": float(iteration),
            "population": float(len(trajectories)),
            "best_cost": float(costs.min()),
            "median_cost": float(costs.median()),
            "markov_entropy": float(entropy),
        })
    assert best is not None
    return TokenSearchResult(
        best[1], best[2], best[0], diagnostics,
        proposed_tokens, transition_evaluations,
    )


def factorized_position_cem(
    initial: Sequence[TokenTrajectory],
    rollout: Callable[[Sequence[Tensor]], Tensor],
    semantic_cost: Callable[[Tensor], Tensor],
    *,
    horizon: int,
    population: int,
    iterations: int,
    elite_fraction: float,
    smoothing: float = 0.1,
    objective: str = "jepa",
    prior_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> TokenSearchResult:
    """Independent-position categorical CEM, retained as a negative control."""

    if min(horizon, population, iterations) < 1:
        raise ValueError("factorized CEM sizes must be positive")
    if not 0 < elite_fraction <= 1 or not 0 <= smoothing < 1:
        raise ValueError("factorized CEM update parameters are invalid")
    initial = _deduplicate(initial)
    if not initial:
        raise ValueError("factorized CEM needs supported trajectories")
    augmented = [_with_stop(item) for item in initial]
    support: list[list[int]] = []
    for position in range(horizon + 1):
        values = sorted({
            row[position] for row in augmented if position < len(row)
        })
        support.append(values)
    probabilities = [
        torch.full((len(values),), 1.0 / len(values)) if values
        else torch.empty(0)
        for values in support
    ]
    best: tuple[float, TokenTrajectory, Tensor] | None = None
    diagnostics: list[dict[str, float]] = []
    proposed_tokens = transition_evaluations = 0
    for iteration in range(iterations):
        sampled: list[TokenTrajectory] = []
        for _ in range(population):
            values: list[int] = []
            logp = 0.0
            terminal = False
            complete = False
            for position in range(horizon + 1):
                if not support[position]:
                    break
                probs = probabilities[position]
                selected = int(torch.multinomial(
                    probs, 1, generator=generator
                ))
                token = support[position][selected]
                logp += float(probs[selected].clamp_min(1e-12).log())
                if token in {_STOP_NEWLINE, _STOP_EOS}:
                    complete = bool(values)
                    terminal = token == _STOP_EOS
                    break
                values.append(token)
            if complete:
                sampled.append(TokenTrajectory(
                    torch.tensor(values, dtype=torch.long), terminal, logp
                ))
        trajectories = _deduplicate(sampled)
        if not trajectories:
            raise RuntimeError("factorized CEM produced no complete trajectory")
        endpoints = rollout([item.tokens for item in trajectories])
        semantic = semantic_cost(endpoints)
        logp = semantic.new_tensor([
            item.log_probability for item in trajectories
        ])
        costs = _score(
            semantic, logp, objective=objective, prior_weight=prior_weight
        )
        proposed_tokens += sum(len(item.tokens) for item in trajectories)
        transition_evaluations += sum(
            len(item.tokens) for item in trajectories
        )
        elite_count = max(1, round(population * elite_fraction))
        elite_ids = costs.topk(
            min(elite_count, len(trajectories)), largest=False
        ).indices.tolist()
        counts = [torch.full((len(values),), 1e-3) for values in support]
        for index in elite_ids:
            row = _with_stop(trajectories[index])
            for position, token in enumerate(row):
                if position >= len(support) or token not in support[position]:
                    continue
                counts[position][support[position].index(token)] += 1
        for position, probs in enumerate(probabilities):
            if not len(probs):
                continue
            updated = counts[position] / counts[position].sum()
            probabilities[position] = (
                smoothing * probs + (1 - smoothing) * updated
            )
        index = int(costs.argmin())
        value = float(costs[index])
        if best is None or value < best[0]:
            best = (
                value, trajectories[index], endpoints[index].detach().clone()
            )
        entropy = torch.stack([
            -(prob.clamp_min(1e-12).log() * prob).sum()
            for prob in probabilities if len(prob)
        ]).mean()
        diagnostics.append({
            "iteration": float(iteration),
            "population": float(len(trajectories)),
            "best_cost": float(costs.min()),
            "median_cost": float(costs.median()),
            "factorized_entropy": float(entropy),
        })
    assert best is not None
    return TokenSearchResult(
        best[1], best[2], best[0], diagnostics,
        proposed_tokens, transition_evaluations,
    )
