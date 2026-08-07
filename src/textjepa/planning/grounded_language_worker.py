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
from textjepa.planning.autoregressive_token_search import (
    TokenTrajectory,
    elite_prefix_cem,
    factorized_position_cem,
    first_order_markov_cem,
    jepa_beam_search,
)
from textjepa.data.language_planning import (
    GENERATION_EOS_TOKEN_IDS,
    STEP_DELIMITER,
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
    search_algorithm: str = "one_shot"
    search_diagnostics: tuple[dict[str, float], ...] = ()
    proposed_tokens: int = 0
    transition_evaluations: int = 0


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
        proposed_tokens=sum(len(tokens) for tokens, _ in candidates),
        transition_evaluations=sum(len(tokens) for tokens, _ in candidates),
    )


def _root_token_history(
    model: HierarchicalLanguageJEPA,
    hidden: Tensor,
    prefix: Tensor,
    prompt_len: int,
) -> tuple[Tensor, Tensor]:
    dtype = next(model.parameters()).dtype
    root = len(prefix)
    first = max(prompt_len, root - model.config.token_context + 1)
    prefix_lengths = torch.arange(first, root + 1, device=prefix.device)
    states = model.e0(hidden[prefix_lengths - 1].to(dtype=dtype))[None]
    actions = model.token_action(prefix[first:root])[None]
    return states, actions


@torch.no_grad()
def _rollout_token_endpoints(
    model: HierarchicalLanguageJEPA,
    state_history: Tensor,
    action_history: Tensor,
    trajectories: list[Tensor] | tuple[Tensor, ...],
) -> Tensor:
    """Roll variable-length branches through P0 with their true root cache."""

    if not trajectories:
        raise ValueError("token rollout requires nonempty trajectories")
    outputs: list[Tensor | None] = [None] * len(trajectories)
    by_length: dict[int, list[int]] = {}
    for index, tokens in enumerate(trajectories):
        if tokens.ndim != 1 or len(tokens) < 1:
            raise ValueError("token rollout trajectories must be nonempty")
        by_length.setdefault(len(tokens), []).append(index)
    for length, indices in by_length.items():
        tokens = torch.stack([
            trajectories[index].to(state_history.device) for index in indices
        ])
        action = model.token_action(tokens)
        rollout, _ = model.p0.rollout(
            state_history[0, -1], action,
            state_history=state_history,
            action_history=action_history,
        )
        coarse = model.e0_to_1(rollout[:, -1])
        for local, index in enumerate(indices):
            outputs[index] = coarse[local]
    assert all(output is not None for output in outputs)
    return torch.stack([output for output in outputs if output is not None])


def _worker_semantic_cost(metric, waypoint: Tensor):
    def score(endpoints: Tensor) -> Tensor:
        value = metric(endpoints, waypoint)
        if value.shape != (len(endpoints),):
            raise ValueError("worker metric must return one cost per endpoint")
        return value
    return score


def _completion(tokenizer):
    newline = tokenizer.encode(STEP_DELIMITER, add_special_tokens=False)

    def complete(tokens: Tensor) -> tuple[bool, bool, Tensor]:
        values = tokens.tolist()
        if values[-1] in GENERATION_EOS_TOKEN_IDS:
            return True, True, tokens[:-1]
        if newline and values[-len(newline):] == newline:
            return True, False, tokens
        return False, False, tokens
    return complete


@torch.no_grad()
def build_optimized_worker_bank(
    model: HierarchicalLanguageJEPA,
    frozen_model,
    tokenizer,
    prefix: Tensor,
    hidden: Tensor,
    requested_waypoint: Tensor,
    metric: Callable[[Tensor, Tensor], Tensor],
    *,
    prompt_len: int,
    algorithm: str,
    objective: str,
    population: int,
    k0: int,
    iterations: int,
    elite_fraction: float,
    beam_width: int,
    branch_factor: int,
    preserve_prefix: int,
    prior_weight: float,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> WorkerBank:
    """Actively optimize a complete token action toward a coarse waypoint."""

    if algorithm not in {
        "beam", "autoregressive_cem", "markov_cem", "factorized_cem"
    }:
        raise ValueError("unknown optimized worker algorithm")
    if algorithm in {"markov_cem", "factorized_cem"} and objective != "jepa":
        raise ValueError(
            "Markov/factorized controls expose proposal probabilities that "
            "are not Qwen likelihoods; evaluate them with JEPA-only cost"
        )
    state_history, action_history = _root_token_history(
        model, hidden, prefix, prompt_len
    )
    rollout_seconds = 0.0

    def rollout(trajectories):
        nonlocal rollout_seconds
        started = perf_counter()
        result = _rollout_token_endpoints(
            model, state_history, action_history, list(trajectories)
        )
        rollout_seconds += perf_counter() - started
        return result

    semantic = _worker_semantic_cost(metric, requested_waypoint)
    generation_started = perf_counter()
    if algorithm == "beam":
        def propose(prefixes, branch):
            if not prefixes:
                raise ValueError("beam proposal received no prefixes")
            width = len(prefixes[0])
            if any(len(item) != width for item in prefixes):
                raise ValueError("beam prefixes must have equal depth")
            suffix = (
                torch.stack([item.to(prefix.device) for item in prefixes])
                if width else prefix.new_empty((len(prefixes), 0))
            )
            full = torch.cat([
                prefix[None].expand(len(prefixes), -1), suffix
            ], -1)
            output = frozen_model(
                input_ids=full,
                attention_mask=torch.ones_like(full, dtype=torch.bool),
                use_cache=False, return_dict=True,
            )
            logp = torch.log_softmax(output.logits[:, -1].float(), -1)
            values, ids = logp.topk(branch, -1)
            return ids.cpu(), values.cpu()

        result = jepa_beam_search(
            propose, rollout, semantic, _completion(tokenizer),
            horizon=k0, beam_width=beam_width,
            branch_factor=branch_factor, objective=objective,
            prior_weight=prior_weight,
        )
    else:
        def generated(
            forced_prefixes: list[Tensor] | None,
            count: int,
            iteration: int,
        ) -> list[TokenTrajectory]:
            groups = forced_prefixes or [prefix.new_empty(0)]
            per_group = max(1, (count + len(groups) - 1) // len(groups))
            collected: list[TokenTrajectory] = []
            for group_index, forced_cpu in enumerate(groups):
                forced = forced_cpu.to(prefix.device)
                remaining = k0 - len(forced)
                if remaining < 1:
                    continue
                try:
                    suffixes = generate_complete_reasoning_candidates(
                        frozen_model, tokenizer, torch.cat([prefix, forced]),
                        population=per_group, max_tokens=remaining,
                        temperature=temperature, top_p=top_p, top_k=top_k,
                        seed=(
                            seed + 1009 * iteration + 9176 * group_index
                        ),
                    )
                except RuntimeError:
                    continue
                for suffix, terminal in suffixes:
                    tokens = torch.cat([forced.cpu(), suffix.cpu()])
                    collected.append(TokenTrajectory(tokens, terminal, 0.0))
            collected = collected[:count]
            if objective != "jepa" and collected:
                grounded = exact_ground_sentence_candidates(
                    frozen_model, prefix,
                    [(item.tokens, item.terminal) for item in collected],
                )
                collected = [
                    TokenTrajectory(
                        item.tokens, item.terminal,
                        float(grounded.log_probabilities[index]),
                    )
                    for index, item in enumerate(collected)
                ]
            return collected

        if algorithm == "autoregressive_cem":
            result = elite_prefix_cem(
                generated, rollout, semantic, population=population,
                iterations=iterations, elite_fraction=elite_fraction,
                preserve_prefix=preserve_prefix, objective=objective,
                prior_weight=prior_weight,
            )
        elif algorithm == "markov_cem":
            initial = generated(None, population, 0)
            result = first_order_markov_cem(
                initial, rollout, semantic, horizon=k0,
                population=population, iterations=iterations,
                elite_fraction=elite_fraction, objective=objective,
                prior_weight=prior_weight,
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            initial = generated(None, population, 0)
            result = factorized_position_cem(
                initial, rollout, semantic, horizon=k0,
                population=population, iterations=iterations,
                elite_fraction=elite_fraction, objective=objective,
                prior_weight=prior_weight,
                generator=torch.Generator().manual_seed(seed),
            )
    generation_seconds = perf_counter() - generation_started - rollout_seconds
    candidate = result.trajectory
    exact_started = perf_counter()
    grounded = exact_ground_sentence_candidates(
        frozen_model, prefix, [(candidate.tokens, candidate.terminal)]
    )
    exact_seconds = perf_counter() - exact_started
    dtype = next(model.parameters()).dtype
    exact_coarse = model.e0_to_1(model.e0(
        grounded.endpoint_hidden.to(dtype=dtype)
    ))
    actions = model.a1(grounded.tokens, grounded.mask)
    return WorkerBank(
        candidates=[(candidate.tokens, candidate.terminal)],
        predicted_coarse=result.predicted_endpoint[None],
        exact_coarse=exact_coarse,
        actions=actions,
        lm_log_probability=grounded.log_probabilities,
        generation_seconds=max(0.0, generation_seconds),
        exact_grounding_seconds=exact_seconds,
        token_rollout_seconds=rollout_seconds,
        search_algorithm=algorithm,
        search_diagnostics=tuple(result.diagnostics),
        proposed_tokens=result.proposed_tokens,
        transition_evaluations=result.transition_evaluations,
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
    objective: str = "combined",
) -> AchievedTransition:
    """Select supported text and return its exact causal successor."""

    if requested_waypoint.shape != (model.config.d_sentence,):
        raise ValueError("requested waypoint has the wrong dimension")
    semantic = metric(bank.predicted_coarse, requested_waypoint)
    if objective == "jepa":
        cost = semantic
    elif objective == "lm":
        cost = -bank.lm_log_probability
    elif objective == "combined":
        cost = semantic - worker_prior_weight * bank.lm_log_probability
    else:
        raise ValueError("unknown worker objective")
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
