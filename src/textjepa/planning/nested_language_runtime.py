r"""Faithful contextual rollouts for the strict nested language hierarchy.

The generic planning helpers operate on tensor-valued Markov states.  The
sentence predictor used by the language hierarchy is different: its causal
state is a bounded history of sentence states and macro actions.  This module
keeps that history explicit so oracle teachers, CEM, and value evaluation do
not silently discard :math:`\kappa^1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from textjepa.models.hierarchical_language_jepa import HierarchicalLanguageJEPA


@dataclass(frozen=True)
class SentencePlanningState:
    """A batched sentence state together with its bounded causal history."""

    state_history: Tensor
    action_history: Tensor
    context: Tensor

    @property
    def state(self) -> Tensor:
        return self.state_history[:, -1]


def _validate_history(
    model: HierarchicalLanguageJEPA,
    state_history: Tensor,
    action_history: Tensor,
) -> None:
    if state_history.ndim != 3 or (
        state_history.shape[-1] != model.config.d_sentence
    ):
        raise ValueError("sentence state history must be [B,L,d_sentence]")
    if action_history.ndim != 3 or (
        action_history.shape[-1] != model.config.d_action
    ):
        raise ValueError("sentence action history must be [B,L-1,d_action]")
    if state_history.shape[0] != action_history.shape[0] or (
        state_history.shape[1] != action_history.shape[1] + 1
    ):
        raise ValueError("sentence histories are not causally aligned")
    if state_history.shape[1] > model.config.sentence_context:
        raise ValueError("sentence history exceeds the configured context")


def sentence_pre_action_context(
    model: HierarchicalLanguageJEPA,
    state_history: Tensor,
    action_history: Tensor,
) -> Tensor:
    """Return ``c_j`` without exposing the current action ``u_j``.

    ``ContextualControlledPredictor`` shifts actions internally, hence its
    context at the last state item is independent of the dummy action placed
    at that position.  The dummy is used only to satisfy rectangular shapes.
    """

    _validate_history(model, state_history, action_history)
    dummy = action_history.new_zeros(
        action_history.shape[0], 1, model.config.d_action
    )
    aligned_actions = torch.cat([action_history, dummy], dim=1)
    valid = torch.ones(
        state_history.shape[:2], dtype=torch.bool,
        device=state_history.device,
    )
    _, context = model.p1(
        state_history, aligned_actions, valid, return_context=True
    )
    return context[:, -1]


def make_sentence_planning_state(
    model: HierarchicalLanguageJEPA,
    state_history: Tensor,
    action_history: Tensor,
) -> SentencePlanningState:
    """Construct the complete pre-action planning state ``xi=(z,kappa)``."""

    return SentencePlanningState(
        state_history=state_history,
        action_history=action_history,
        context=sentence_pre_action_context(
            model, state_history, action_history
        ),
    )


def sentence_planning_state_from_trace(
    model: HierarchicalLanguageJEPA,
    hidden: Tensor,
    token_ids: Tensor,
    boundaries: Tensor,
    boundary_index: int,
) -> SentencePlanningState:
    """Rebuild an exact bounded sentence history at a real trace boundary.

    ``boundaries`` are global prefix lengths. The returned state is rooted at
    ``H(boundaries[boundary_index])`` and every preceding action is encoded
    from the exact Python slice ``token_ids[B_j:B_{j+1}]``.
    """

    if hidden.ndim != 2 or token_ids.ndim != 1 or boundaries.ndim != 1:
        raise ValueError("trace tensors must be unbatched")
    if hidden.shape[0] != token_ids.shape[0] or (
        hidden.shape[-1] != model.config.d_backbone
    ):
        raise ValueError("trace hidden states and token IDs do not align")
    if boundary_index < 0 or boundary_index >= len(boundaries):
        raise ValueError("boundary index lies outside the trace")
    if bool((boundaries[1:] <= boundaries[:-1]).any()) or (
        int(boundaries[0]) < 1 or int(boundaries[-1]) > len(token_ids)
    ):
        raise ValueError("global prefix boundaries are invalid")
    begin = max(
        0, boundary_index - model.config.sentence_context + 1
    )
    chosen = boundaries[begin:boundary_index + 1]
    boundary_hidden = hidden[chosen - 1]
    states = model.encode_sentence(boundary_hidden)[None]
    spans = []
    for index in range(begin, boundary_index):
        spans.append(token_ids[boundaries[index]:boundaries[index + 1]])
    if spans:
        width = max(len(span) for span in spans)
        if width > model.config.max_span:
            raise ValueError("trace sentence exceeds the action encoder cap")
        ids = token_ids.new_full(
            (len(spans), width), model.config.pad_id
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, span in enumerate(spans):
            ids[row, :len(span)] = span
            mask[row, :len(span)] = True
        actions = model.a1(ids, mask)[None]
    else:
        actions = states.new_empty(1, 0, model.config.d_action)
    return make_sentence_planning_state(model, states, actions)


def repeat_sentence_planning_state(
    planning_state: SentencePlanningState, repeats: int
) -> SentencePlanningState:
    if repeats < 1 or planning_state.state_history.shape[0] != 1:
        raise ValueError("repeat requires one root and a positive population")
    return SentencePlanningState(
        planning_state.state_history.expand(repeats, -1, -1).clone(),
        planning_state.action_history.expand(repeats, -1, -1).clone(),
        planning_state.context.expand(repeats, -1).clone(),
    )


def advance_sentence_planning_state(
    model: HierarchicalLanguageJEPA,
    planning_state: SentencePlanningState,
    action: Tensor,
) -> SentencePlanningState:
    """Apply one action and return the successor's pre-action context."""

    _validate_history(
        model, planning_state.state_history, planning_state.action_history
    )
    if action.shape != (
        planning_state.state_history.shape[0], model.config.d_action
    ):
        raise ValueError("macro action does not align with planning branches")
    aligned_actions = torch.cat(
        [planning_state.action_history, action[:, None]], dim=1
    )
    valid = torch.ones(
        planning_state.state_history.shape[:2], dtype=torch.bool,
        device=planning_state.state_history.device,
    )
    prediction, _ = model.p1(
        planning_state.state_history, aligned_actions, valid,
        return_context=True,
    )
    successor = prediction[:, -1]
    states = torch.cat(
        [planning_state.state_history, successor[:, None]], dim=1
    )
    actions = aligned_actions
    keep = model.config.sentence_context
    if states.shape[1] > keep:
        states = states[:, -keep:]
        actions = actions[:, -(keep - 1):] if keep > 1 else actions[:, :0]
    return make_sentence_planning_state(model, states, actions)


def append_exact_sentence_transition(
    model: HierarchicalLanguageJEPA,
    planning_state: SentencePlanningState,
    action: Tensor,
    exact_successor: Tensor,
) -> SentencePlanningState:
    """Advance the cache with a worker-achieved, exactly encoded successor."""

    _validate_history(
        model, planning_state.state_history, planning_state.action_history
    )
    batch = planning_state.state_history.shape[0]
    if action.shape != (batch, model.config.d_action) or (
        exact_successor.shape != (batch, model.config.d_sentence)
    ):
        raise ValueError("exact sentence transition does not align with branches")
    states = torch.cat(
        [planning_state.state_history, exact_successor[:, None]], 1
    )
    actions = torch.cat(
        [planning_state.action_history, action[:, None]], 1
    )
    keep = model.config.sentence_context
    if states.shape[1] > keep:
        states = states[:, -keep:]
        actions = actions[:, -(keep - 1):] if keep > 1 else actions[:, :0]
    return make_sentence_planning_state(model, states, actions)


@dataclass(frozen=True)
class MacroRollout:
    actions: Tensor
    states: Tensor
    contexts: Tensor
    log_probabilities: Tensor
    final_planning_state: SentencePlanningState


def rollout_prior_noise(
    model: HierarchicalLanguageJEPA,
    initial: SentencePlanningState,
    task: Tensor,
    noise: Tensor,
) -> MacroRollout:
    """Decode prior-coordinate noise while preserving every branch cache."""

    if model.pi1 is None:
        raise ValueError("macro rollouts require Pi1")
    if noise.ndim != 3 or noise.shape[-1] != model.config.d_action:
        raise ValueError("noise must be [population,horizon,d_action]")
    population, horizon, _ = noise.shape
    if horizon < 1:
        raise ValueError("macro rollout horizon must be positive")
    state = repeat_sentence_planning_state(initial, population)
    if task.ndim == 1:
        task = task[None].expand(population, -1)
    if task.shape != (population, model.config.d_task):
        raise ValueError("task embedding does not align with population")
    actions, states, contexts, log_probabilities = [], [], [], []
    log2pi = noise.new_tensor(2 * torch.pi).log()
    for step in range(horizon):
        mean, logvar = model.pi1.prior_params(
            state.state, state.context, task
        )
        action = mean + (0.5 * logvar).exp() * noise[:, step]
        log_probability = -0.5 * (
            noise[:, step].square() + logvar + log2pi
        ).sum(-1)
        state = advance_sentence_planning_state(model, state, action)
        actions.append(action)
        states.append(state.state)
        contexts.append(state.context)
        log_probabilities.append(log_probability)
    return MacroRollout(
        actions=torch.stack(actions, 1),
        states=torch.stack(states, 1),
        contexts=torch.stack(contexts, 1),
        log_probabilities=torch.stack(log_probabilities, 1),
        final_planning_state=state,
    )


def rollout_macro_actions(
    model: HierarchicalLanguageJEPA,
    initial: SentencePlanningState,
    task: Tensor,
    actions: Tensor,
) -> MacroRollout:
    """Roll ambient macro actions while preserving every contextual branch.

    This is the explicit no-prior search ablation.  Pi1 is used only to record
    diagnostic log probabilities; it does not transform or constrain actions.
    """

    if actions.ndim != 3 or actions.shape[-1] != model.config.d_action:
        raise ValueError("actions must be [population,horizon,d_action]")
    population, horizon, _ = actions.shape
    if horizon < 1:
        raise ValueError("macro rollout horizon must be positive")
    state = repeat_sentence_planning_state(initial, population)
    if task.ndim == 1:
        task = task[None].expand(population, -1)
    if task.shape != (population, model.config.d_task):
        raise ValueError("task embedding does not align with population")
    states, contexts, log_probabilities = [], [], []
    for step in range(horizon):
        action = actions[:, step]
        if model.pi1 is None:
            logp = action.new_zeros(population)
        else:
            logp = macro_action_log_probability(model, state, task, action)
        state = advance_sentence_planning_state(model, state, action)
        states.append(state.state)
        contexts.append(state.context)
        log_probabilities.append(logp)
    return MacroRollout(
        actions=actions,
        states=torch.stack(states, 1),
        contexts=torch.stack(contexts, 1),
        log_probabilities=torch.stack(log_probabilities, 1),
        final_planning_state=state,
    )


def macro_action_log_probability(
    model: HierarchicalLanguageJEPA,
    planning_state: SentencePlanningState,
    task: Tensor,
    action: Tensor,
) -> Tensor:
    """Score an achieved action under ``Pi1`` at its pre-action state."""

    if model.pi1 is None:
        raise ValueError("macro-action scoring requires Pi1")
    batch = planning_state.state.shape[0]
    if task.ndim == 1:
        task = task[None].expand(batch, -1)
    if task.shape != (batch, model.config.d_task) or action.shape != (
        batch, model.config.d_action
    ):
        raise ValueError("macro-action score inputs do not align")
    mean, logvar = model.pi1.prior_params(
        planning_state.state, planning_state.context, task
    )
    standardized = (action - mean) * (-0.5 * logvar).exp()
    log2pi = action.new_tensor(2 * torch.pi).log()
    return -0.5 * (standardized.square() + logvar + log2pi).sum(-1)


@dataclass(frozen=True)
class ContextualCEMResult:
    noise: Tensor
    rollout: MacroRollout
    selected_prefix: int
    cost: float
    diagnostics: list[dict[str, float]]


def contextual_prior_cem(
    model: HierarchicalLanguageJEPA,
    initial: SentencePlanningState,
    task: Tensor,
    objective: Callable[[MacroRollout], tuple[Tensor, Tensor]],
    *,
    horizon: int,
    population: int,
    iterations: int,
    elite_fraction: float = 0.1,
    smoothing: float = 0.1,
    covariance_floor: float = 0.05,
    trust_region: float = 3.0,
    ground: Callable[[Tensor, MacroRollout, Tensor], Tensor] | None = None,
    ground_topn: int = 0,
    ground_random: int = 0,
    select_grounded: bool = False,
    generator: torch.Generator | None = None,
) -> ContextualCEMResult:
    """CEM in ``Pi1`` coordinates over complete contextual sentence states.

    The objective returns ``(plan_cost, selected_prefix_index)`` for every
    population member. Prefix indices are zero based, so zero means a
    one-action plan.
    """

    if min(horizon, population, iterations) < 1:
        raise ValueError("CEM sizes must be positive")
    if not 0 < elite_fraction <= 1 or not 0 <= smoothing < 1:
        raise ValueError("invalid CEM update configuration")
    if covariance_floor <= 0 or trust_region < 0:
        raise ValueError("invalid CEM support constraint")
    if min(ground_topn, ground_random) < 0:
        raise ValueError("grounding counts must be nonnegative")
    if select_grounded and (
        ground is None or ground_topn + ground_random < 1
    ):
        raise ValueError("grounded CEM selection requires grounded candidates")
    mean = initial.state.new_zeros(horizon, model.config.d_action)
    std = initial.state.new_ones(horizon, model.config.d_action)
    elite_count = max(1, round(population * elite_fraction))
    best: tuple[float, Tensor, MacroRollout, int] | None = None
    diagnostics: list[dict[str, float]] = []
    for iteration in range(iterations):
        noise = mean + std * torch.randn(
            population, horizon, model.config.d_action,
            device=mean.device, dtype=mean.dtype, generator=generator,
        )
        rollout = rollout_prior_noise(model, initial, task, noise)
        cost, prefix = objective(rollout)
        if cost.shape != (population,) or prefix.shape != (population,):
            raise ValueError("CEM objective must return aligned vectors")
        support_norm = noise.square().mean((-1, -2)).sqrt()
        support_penalty = (
            support_norm - trust_region
        ).clamp_min(0).square()
        cost = cost + support_penalty
        selection_cost = cost
        grounded_ids = None
        grounded_cost = None
        if ground is not None and (ground_topn or ground_random):
            top = cost.topk(min(ground_topn, population), largest=False).indices
            available = torch.ones(
                population, dtype=torch.bool, device=cost.device
            )
            available[top] = False
            pool = available.nonzero().flatten()
            random_count = min(ground_random, len(pool))
            random_ids = (
                pool[torch.randperm(
                    len(pool), device=pool.device, generator=generator
                )[:random_count]] if random_count else top[:0]
            )
            grounded_ids = torch.unique(torch.cat([top, random_ids]))
            grounded_raw_cost = ground(
                noise[grounded_ids], rollout, grounded_ids
            )
            if grounded_raw_cost.shape != (len(grounded_ids),) or not bool(
                torch.isfinite(grounded_raw_cost).all()
            ):
                raise ValueError("ground callback must return finite scalar costs")
            grounded_cost = (
                grounded_raw_cost + support_penalty[grounded_ids]
            )
            if select_grounded:
                selection_cost = torch.full_like(cost, torch.inf)
                selection_cost[grounded_ids] = grounded_cost
        if select_grounded and grounded_ids is not None:
            count = min(elite_count, len(grounded_ids))
            local = grounded_cost.topk(count, largest=False).indices
            elite_ids = grounded_ids[local]
        else:
            elite_ids = cost.topk(elite_count, largest=False).indices
        elite = noise[elite_ids]
        new_mean = elite.mean(0)
        new_std = elite.std(0, unbiased=False).clamp_min(covariance_floor)
        mean = smoothing * mean + (1 - smoothing) * new_mean
        std = smoothing * std + (1 - smoothing) * new_std
        selected = int(selection_cost.argmin())
        selected_cost = float(selection_cost[selected])
        if best is None or selected_cost < best[0]:
            # Keep a single-branch rollout with the exact winning history.
            winner = rollout_prior_noise(
                model, initial, task, noise[selected:selected + 1]
            )
            best = (
                selected_cost, noise[selected].clone(), winner,
                int(prefix[selected]),
            )
        optimizer_regret = float("nan")
        if grounded_ids is not None and grounded_cost is not None:
            predicted_best = int(cost.argmin())
            match = (grounded_ids == predicted_best).nonzero().flatten()
            if len(match):
                optimizer_regret = float(
                    grounded_cost[int(match[0])] - grounded_cost.min()
                )
        diagnostics.append({
            "iteration": float(iteration),
            "best_predicted_cost": float(cost.min()),
            "median_predicted_cost": float(cost.median()),
            "elite_noise_norm": float(support_norm[elite_ids].mean()),
            "cem_std": float(std.mean()),
            "unique_plans": float(torch.unique(
                rollout.actions.detach(), dim=0
            ).shape[0]),
            "best_grounded_cost": (
                float(grounded_cost.min())
                if grounded_cost is not None else float("nan")
            ),
            "grounded_candidates": float(
                len(grounded_ids) if grounded_ids is not None else 0
            ),
            "optimizer_curse_regret": optimizer_regret,
        })
    assert best is not None
    return ContextualCEMResult(
        noise=best[1], rollout=best[2], selected_prefix=best[3],
        cost=best[0], diagnostics=diagnostics,
    )


def contextual_action_cem(
    model: HierarchicalLanguageJEPA,
    initial: SentencePlanningState,
    task: Tensor,
    objective: Callable[[MacroRollout], tuple[Tensor, Tensor]],
    *,
    horizon: int,
    population: int,
    iterations: int,
    elite_fraction: float = 0.1,
    smoothing: float = 0.1,
    covariance_floor: float = 0.05,
    generator: torch.Generator | None = None,
) -> ContextualCEMResult:
    """Unconstrained Gaussian CEM over ambient sentence-action vectors."""

    if min(horizon, population, iterations) < 1:
        raise ValueError("CEM sizes must be positive")
    if not 0 < elite_fraction <= 1 or not 0 <= smoothing < 1:
        raise ValueError("invalid CEM update configuration")
    if covariance_floor <= 0:
        raise ValueError("CEM covariance floor must be positive")
    mean = initial.state.new_zeros(horizon, model.config.d_action)
    std = initial.state.new_ones(horizon, model.config.d_action)
    elite_count = max(1, round(population * elite_fraction))
    best: tuple[float, Tensor, MacroRollout, int] | None = None
    diagnostics: list[dict[str, float]] = []
    for iteration in range(iterations):
        actions = mean + std * torch.randn(
            population, horizon, model.config.d_action,
            device=mean.device, dtype=mean.dtype, generator=generator,
        )
        rollout = rollout_macro_actions(model, initial, task, actions)
        cost, prefix = objective(rollout)
        if cost.shape != (population,) or prefix.shape != (population,):
            raise ValueError("CEM objective must return aligned vectors")
        if not bool(torch.isfinite(cost).all()):
            raise ValueError("ambient CEM objective must be finite")
        elite_ids = cost.topk(elite_count, largest=False).indices
        elite = actions[elite_ids]
        new_mean = elite.mean(0)
        new_std = elite.std(0, unbiased=False).clamp_min(covariance_floor)
        mean = smoothing * mean + (1 - smoothing) * new_mean
        std = smoothing * std + (1 - smoothing) * new_std
        selected = int(cost.argmin())
        selected_cost = float(cost[selected])
        if best is None or selected_cost < best[0]:
            winner = rollout_macro_actions(
                model, initial, task, actions[selected:selected + 1]
            )
            best = (
                selected_cost, actions[selected].clone(), winner,
                int(prefix[selected]),
            )
        diagnostics.append({
            "iteration": float(iteration),
            "best_predicted_cost": float(cost.min()),
            "median_predicted_cost": float(cost.median()),
            "elite_action_norm": float(elite.square().mean((-1, -2)).sqrt().mean()),
            "cem_std": float(std.mean()),
            "unique_plans": float(torch.unique(
                rollout.actions.detach(), dim=0
            ).shape[0]),
            "best_grounded_cost": float("nan"),
            "grounded_candidates": 0.0,
            "optimizer_curse_regret": float("nan"),
        })
    assert best is not None
    return ContextualCEMResult(
        noise=best[1], rollout=best[2], selected_prefix=best[3],
        cost=best[0], diagnostics=diagnostics,
    )
