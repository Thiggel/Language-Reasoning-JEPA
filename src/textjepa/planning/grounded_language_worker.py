"""Worker execution and exact grounding shared by MPC and value teachers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor

from textjepa.models.hierarchical_language_jepa import HierarchicalLanguageJEPA
from textjepa.planning.nested_language_runtime import (
    SentencePlanningState,
    append_exact_sentence_transition,
    macro_action_log_probability,
)
from textjepa.utils.hierarchical_generation import (
    generate_complete_reasoning_candidates,
    exact_ground_sentence_candidates,
)


@dataclass(frozen=True)
class WorkerBank:
    candidates: list[tuple[Tensor, bool]]
    predicted_coarse: Tensor
    exact_coarse: Tensor
    actions: Tensor
    lm_log_probability: Tensor
    generation_seconds: float
    exact_grounding_seconds: float
    token_rollout_seconds: float
    worker_scoring_seconds: float = 0.0


@dataclass(frozen=True)
class AchievedTransition:
    planning_state: SentencePlanningState
    tokens: Tensor
    terminal: bool
    action: Tensor
    prior_log_probability: Tensor
    selected_index: int
    waypoint_cost: float


@torch.no_grad()
def encode_frozen_prefix(frozen_model, tokens: Tensor) -> Tensor:
    output = frozen_model(
        input_ids=tokens[None],
        attention_mask=torch.ones_like(tokens[None], dtype=torch.bool),
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    return output.hidden_states[-1][0]


@torch.no_grad()
def build_worker_bank(
    model: HierarchicalLanguageJEPA,
    frozen_model,
    tokenizer,
    prefix: Tensor,
    hidden: Tensor,
    *,
    prompt_len: int,
    population: int,
    k0: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> WorkerBank:
    """Generate text, predict P0 endpoints, and exactly re-encode them."""

    started = perf_counter()
    candidates = generate_complete_reasoning_candidates(
        frozen_model, tokenizer, prefix, population=population,
        max_tokens=k0, temperature=temperature, top_p=top_p, top_k=top_k,
        seed=seed,
    )
    generation_seconds = perf_counter() - started
    started = perf_counter()
    grounded = exact_ground_sentence_candidates(
        frozen_model, prefix, candidates
    )
    exact_grounding_seconds = perf_counter() - started
    dtype = next(model.parameters()).dtype
    started = perf_counter()
    root = len(prefix)
    first = max(prompt_len, root - model.config.token_context + 1)
    prefix_lengths = torch.arange(first, root + 1, device=prefix.device)
    state_history = model.e0(hidden[prefix_lengths - 1].to(dtype=dtype))[None]
    action_history = model.token_action(prefix[first:root])[None]
    predicted = []
    for candidate, _ in candidates:
        action = model.token_action(candidate.to(prefix.device))[None]
        rollout, _ = model.p0.rollout(
            state_history[0, -1], action,
            state_history=state_history, action_history=action_history,
        )
        predicted.append(rollout[0, -1])
    predicted_coarse = model.e0_to_1(torch.stack(predicted))
    token_rollout_seconds = perf_counter() - started
    exact_coarse = model.e0_to_1(model.e0(
        grounded.endpoint_hidden.to(dtype=dtype)
    ))
    actions = model.a1(grounded.tokens, grounded.mask)
    return WorkerBank(
        candidates=candidates,
        predicted_coarse=predicted_coarse,
        exact_coarse=exact_coarse,
        actions=actions,
        lm_log_probability=grounded.log_probabilities,
        generation_seconds=generation_seconds,
        exact_grounding_seconds=exact_grounding_seconds,
        token_rollout_seconds=token_rollout_seconds,
    )


@torch.no_grad()
def realize_macro_action(
    model: HierarchicalLanguageJEPA,
    planning_state: SentencePlanningState,
    task: Tensor,
    requested_waypoint: Tensor,
    bank: WorkerBank,
    metric: Callable[[Tensor, Tensor], Tensor],
    *,
    worker_prior_weight: float,
) -> AchievedTransition:
    """Select supported text and return its exact causal successor."""

    if requested_waypoint.shape != (model.config.d_sentence,):
        raise ValueError("requested waypoint has the wrong dimension")
    cost = metric(bank.predicted_coarse, requested_waypoint) - (
        worker_prior_weight * bank.lm_log_probability
    )
    selected = int(cost.argmin())
    achieved_action = bank.actions[selected:selected + 1]
    successor = append_exact_sentence_transition(
        model, planning_state, achieved_action,
        bank.exact_coarse[selected:selected + 1],
    )
    prior_logp = macro_action_log_probability(
        model, planning_state, task, achieved_action
    )
    tokens, terminal = bank.candidates[selected]
    return AchievedTransition(
        planning_state=successor, tokens=tokens.to(bank.actions.device),
        terminal=terminal, action=achieved_action,
        prior_log_probability=prior_logp, selected_index=selected,
        waypoint_cost=float(cost[selected]),
    )
