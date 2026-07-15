"""No-LM CEM primitives for hierarchical token world models.

All functions operate on supplied rollout callables, making the optimizer
independent of TextJEPA model details and straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def latent_l1(x: Tensor, y: Tensor) -> Tensor:
    return (
        F.layer_norm(x, x.shape[-1:])
        - F.layer_norm(y, y.shape[-1:])
    ).abs().mean(-1)


@dataclass
class CEMResult:
    actions: Tensor
    states: Tensor
    cost: float
    diagnostics: dict[str, float]


def project_to_bank(actions: Tensor, bank: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest-neighbour projection and pre-projection squared distance."""
    if bank.ndim != 2 or len(bank) == 0 or bank.shape[-1] != actions.shape[-1]:
        raise ValueError("bank must be nonempty [codes, action_dim]")
    flat = actions.reshape(-1, actions.shape[-1])
    distance = torch.cdist(flat, bank).square()
    nearest = distance.argmin(-1)
    projected = bank[nearest].reshape_as(actions)
    return projected, distance.gather(1, nearest[:, None]).reshape(
        actions.shape[:-1]
    ).squeeze(-1)


def conditional_bank(
    state: Tensor, bank_states: Tensor, bank_actions: Tensor, k: int
) -> Tensor:
    """Actions attached to the k nearest observed starting states."""
    if state.dim() != 1:
        raise ValueError("conditional_bank expects one current state")
    k = min(int(k), len(bank_states))
    ids = torch.cdist(state[None], bank_states).squeeze(0).topk(
        k, largest=False
    ).indices
    return bank_actions[ids]


def categorical_cem(
    rollout: Callable[[Tensor], Tensor],
    goal: Tensor,
    horizon: int,
    vocab_size: int,
    candidates: int = 256,
    iterations: int = 5,
    elites: int = 32,
    alpha: float = 0.1,
    forbidden: tuple[int, ...] = (),
    extra_cost: Callable[[Tensor, Tensor], tuple[Tensor, dict[str, Tensor]]] | None = None,
    generator: torch.Generator | None = None,
) -> CEMResult:
    """Categorical CEM over discrete token sequences, with no language model."""
    if horizon < 1 or vocab_size < 2:
        raise ValueError("positive horizon and vocab_size >= 2 required")
    if candidates < 1 or iterations < 1 or elites < 1:
        raise ValueError("candidates, iterations, and elites must be positive")
    device = goal.device
    probs = torch.ones(horizon, vocab_size, device=device)
    if forbidden:
        probs[:, list(forbidden)] = 0
    if bool((probs.sum(-1) == 0).any()):
        raise ValueError("forbidden tokens remove the complete vocabulary")
    probs /= probs.sum(-1, keepdim=True)
    elite_n = min(elites, candidates)
    best_cost, best_tokens, best_states = float("inf"), None, None
    best_diag: dict[str, float] = {}
    for _ in range(iterations):
        tokens = torch.stack([
            torch.multinomial(
                probs[t], candidates, replacement=True, generator=generator
            )
            for t in range(horizon)
        ], 1)
        states = rollout(tokens)
        goal_cost = latent_l1(states[:, -1], goal.expand(candidates, -1))
        cost = goal_cost
        diagnostics: dict[str, Tensor] = {"goal_cost": goal_cost.detach()}
        if extra_cost is not None:
            addition, extra = extra_cost(tokens, states)
            if addition.shape != cost.shape:
                raise ValueError("categorical extra_cost must return one cost per candidate")
            cost = cost + addition
            diagnostics.update(extra)
        ids = cost.topk(elite_n, largest=False).indices
        elite = tokens[ids]
        counts = torch.stack([
            torch.bincount(elite[:, t], minlength=vocab_size)
            for t in range(horizon)
        ]).float()
        if forbidden:
            counts[:, list(forbidden)] = 0
        new_probs = (counts + 1e-3) / (counts.sum(-1, keepdim=True) + 1e-3 * vocab_size)
        probs = alpha * probs + (1 - alpha) * new_probs
        index = int(cost.argmin())
        if float(cost[index]) < best_cost:
            best_cost = float(cost[index])
            best_tokens = tokens[index].clone()
            best_states = states[index].clone()
            best_diag = {
                key: float(value[index])
                for key, value in diagnostics.items()
                if value.ndim == 1 and torch.isfinite(value[index])
            }
    assert best_tokens is not None and best_states is not None
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1).mean()
    return CEMResult(
        best_tokens, best_states, best_cost,
        {**best_diag, "categorical_entropy": float(entropy)},
    )


def batched_categorical_min_cost(
    rollout: Callable[[Tensor], Tensor],
    goals: Tensor,
    horizon: int,
    vocab_size: int,
    candidates: int = 64,
    iterations: int = 2,
    elites: int = 8,
    alpha: float = 0.1,
    forbidden: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
) -> Tensor:
    """Minimum categorical-CEM residual for several subgoals in parallel."""
    groups = len(goals)
    probs = torch.ones(groups, horizon, vocab_size, device=goals.device)
    if forbidden:
        probs[:, :, list(forbidden)] = 0
    if bool((probs.sum(-1) == 0).any()):
        raise ValueError("forbidden tokens remove the complete vocabulary")
    probs /= probs.sum(-1, keepdim=True)
    elite_n = min(elites, candidates)
    best = goals.new_full((groups,), torch.inf)
    for _ in range(iterations):
        tokens = torch.stack([
            torch.multinomial(
                probs[:, step], candidates, replacement=True,
                generator=generator,
            )
            for step in range(horizon)
        ], -1)
        states = rollout(tokens.reshape(groups * candidates, horizon))
        final = states[:, -1].reshape(groups, candidates, -1)
        cost = latent_l1(final, goals[:, None].expand_as(final))
        best = torch.minimum(best, cost.amin(1))
        ids = cost.topk(elite_n, largest=False).indices
        elite = tokens.gather(1, ids[:, :, None].expand(-1, -1, horizon))
        counts = F.one_hot(elite, vocab_size).float().sum(1)
        if forbidden:
            counts[:, :, list(forbidden)] = 0
        new_probs = (counts + 1e-3) / (
            counts.sum(-1, keepdim=True) + 1e-3 * vocab_size
        )
        probs = alpha * probs + (1 - alpha) * new_probs
    return best


def gaussian_mixture_nll(
    actions: Tensor, weights: Tensor, means: Tensor, covariances: Tensor
) -> Tensor:
    """Mean trajectory NLL under a full-covariance Gaussian mixture."""
    flat = actions.reshape(-1, actions.shape[-1])
    terms = []
    for component in range(len(weights)):
        dist = torch.distributions.MultivariateNormal(
            means[component], covariance_matrix=covariances[component]
        )
        terms.append(weights[component].log() + dist.log_prob(flat))
    nll = -torch.logsumexp(torch.stack(terms, -1), -1)
    return nll.reshape(actions.shape[:-1]).mean(-1)


def continuous_cem(
    rollout: Callable[[Tensor], Tensor],
    goal: Tensor,
    horizon: int,
    action_dim: int,
    candidates: int = 256,
    iterations: int = 5,
    elites: int = 32,
    alpha: float = 0.1,
    init_mean: Tensor | None = None,
    init_std: Tensor | None = None,
    project_bank: Tensor | None = None,
    extra_cost: Callable[[Tensor, Tensor], tuple[Tensor, dict[str, Tensor]]] | None = None,
    reachability: Callable[[Tensor], Tensor] | None = None,
    reach_topn: int = 0,
    reach_weight: float = 0.0,
    generator: torch.Generator | None = None,
) -> CEMResult:
    """Gaussian CEM with optional codebook and top-N reachability reranking.

    ``reachability`` receives the first predicted subgoal for each retained
    trajectory and returns one residual per trajectory. Only evaluated top-N
    candidates may become elites, matching the practical HWM refinement.
    """
    if horizon < 1 or action_dim < 1:
        raise ValueError("horizon and action_dim must be positive")
    if candidates < 1 or iterations < 1 or elites < 1:
        raise ValueError("candidates, iterations, and elites must be positive")
    device = goal.device
    mean = torch.zeros(1, horizon, action_dim, device=device)
    std = torch.ones_like(mean)
    if init_mean is not None:
        mean.copy_(init_mean.reshape(1, 1, -1).expand_as(mean))
    if init_std is not None:
        std.copy_(init_std.reshape(1, 1, -1).expand_as(std).clamp_min(0.025))
    elite_n = min(elites, candidates)
    best_cost, best_actions, best_states = float("inf"), None, None
    best_diag: dict[str, float] = {}
    for _ in range(iterations):
        noise = torch.randn(
            candidates, horizon, action_dim, device=device,
            generator=generator,
        )
        raw = mean + std * noise
        projection_distance = raw.new_zeros(candidates)
        actions = raw
        if project_bank is not None:
            actions, distance = project_to_bank(raw, project_bank)
            projection_distance = distance.mean(-1)
        rolled = rollout(actions)
        if isinstance(rolled, tuple):
            states, scored_actions = rolled
        else:
            states, scored_actions = rolled, actions
        cost = latent_l1(states[:, -1], goal.expand(candidates, -1))
        diagnostics: dict[str, Tensor] = {
            "goal_cost": cost.detach(),
            "bank_projection_distance": projection_distance.detach(),
        }
        if extra_cost is not None:
            addition, extra = extra_cost(scored_actions, states)
            cost = cost + addition
            diagnostics.update(extra)
        if reachability is not None and reach_topn > 0 and reach_weight > 0:
            retained_n = min(max(reach_topn, elite_n), candidates)
            retained = cost.topk(retained_n, largest=False).indices
            residual = reachability(states[retained, 0])
            refined = torch.full_like(cost, torch.inf)
            refined[retained] = cost[retained] + reach_weight * residual
            diagnostics["reachability"] = torch.full_like(cost, torch.nan)
            diagnostics["reachability"][retained] = residual.detach()
            cost = refined
        ids = cost.topk(elite_n, largest=False).indices
        elite = raw[ids]  # refit the optimization distribution, not projected codes
        new_mean = elite.mean(0, keepdim=True)
        new_std = elite.std(0, unbiased=False, keepdim=True).clamp_min(0.025)
        mean = alpha * mean + (1 - alpha) * new_mean
        std = alpha * std + (1 - alpha) * new_std
        index = int(cost.argmin())
        if float(cost[index]) < best_cost:
            best_cost = float(cost[index])
            best_actions = scored_actions[index].clone()
            best_states = states[index].clone()
            best_diag = {
                key: float(value[index])
                for key, value in diagnostics.items()
                if value.ndim == 1 and torch.isfinite(value[index])
            }
    assert best_actions is not None and best_states is not None
    best_diag["cem_std"] = float(std.mean())
    return CEMResult(best_actions, best_states, best_cost, best_diag)
