"""Small, deterministic latent tree-search primitives.

The search code knows nothing about text, symbolic feasibility, or a language
model.  Callers provide an on-support action proposal, a recurrent latent
transition, and a leaf cost.  This keeps beam/A* and PUCT comparisons matched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch


Tensor = torch.Tensor


@dataclass
class TreeResult:
    actions: Tensor
    states: Tensor
    cost: float
    diagnostics: dict[str, float]


@dataclass
class _Path:
    actions: Tensor
    states: Tensor
    log_prior: float
    cost: float


def _empty_actions(example: Tensor) -> Tensor:
    return example.new_empty((0,) + tuple(example.shape[1:]))


def best_first_search(
    transition: Callable[[Tensor], Tensor],
    propose: Callable[[Tensor, int], tuple[Tensor, Tensor]],
    leaf_cost: Callable[[Tensor], Tensor],
    start: Tensor,
    action_example: Tensor,
    horizon: int,
    width: int,
    topk: int,
    prior_weight: float = 0.0,
    mode: str = "beam",
) -> TreeResult:
    """Prior-guided beam or bounded A* over a deterministic latent model."""
    if mode not in {"beam", "astar"}:
        raise ValueError("mode must be beam or astar")
    empty = _empty_actions(action_example)
    root = _Path(empty, start.new_empty((0, start.numel())), 0.0, float("inf"))
    frontier = [root]
    completed: list[_Path] = []
    expansions = 0
    while frontier:
        if mode == "beam":
            current, frontier = frontier, []
        else:
            current = [min(frontier, key=lambda path: path.cost)]
            frontier.remove(current[0])
        expanded: list[_Path] = []
        for path in current:
            depth = len(path.actions)
            state = start if depth == 0 else path.states[-1]
            actions, probabilities = propose(state, topk)
            probabilities = probabilities.clamp_min(1e-12)
            for action, probability in zip(actions, probabilities):
                candidate_actions = torch.cat([path.actions, action[None]], 0)
                states = transition(candidate_actions)
                log_prior = path.log_prior + float(probability.log())
                goal = float(leaf_cost(states[-1:])[0])
                score = goal - prior_weight * log_prior / len(candidate_actions)
                child = _Path(candidate_actions, states, log_prior, score)
                expansions += 1
                if len(candidate_actions) >= horizon:
                    completed.append(child)
                else:
                    expanded.append(child)
        if mode == "beam":
            frontier = sorted(expanded, key=lambda path: path.cost)[:width]
            if frontier and len(frontier[0].actions) >= horizon:
                completed.extend(frontier)
                break
            if not frontier and completed:
                break
        else:
            frontier.extend(expanded)
            frontier = sorted(frontier, key=lambda path: path.cost)[:width]
            if len(completed) >= width:
                break
    if not completed:
        raise RuntimeError("tree search did not reach its horizon")
    best = min(completed, key=lambda path: path.cost)
    return TreeResult(best.actions, best.states, best.cost, {
        "tree_expansions": float(expansions),
        "tree_completed": float(len(completed)),
        "tree_prior_nll": float(-best.log_prior / max(1, len(best.actions))),
    })


@dataclass
class _Node:
    actions: Tensor
    states: Tensor
    prior: float
    parent: "_Node | None" = None
    visits: int = 0
    value_sum: float = 0.0
    children: list["_Node"] = field(default_factory=list)
    expanded: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def puct_search(
    transition: Callable[[Tensor], Tensor],
    propose: Callable[[Tensor, int], tuple[Tensor, Tensor]],
    leaf_cost: Callable[[Tensor], Tensor],
    start: Tensor,
    action_example: Tensor,
    horizon: int,
    simulations: int,
    topk: int,
    c_puct: float = 1.5,
    progressive: bool = False,
    widening_c: float = 2.0,
    widening_alpha: float = 0.5,
) -> TreeResult:
    """PUCT with optional progressive widening for supplied proposals."""
    empty = _empty_actions(action_example)
    root = _Node(empty, start.new_empty((0, start.numel())), 1.0)
    best_actions: Tensor | None = None
    best_states: Tensor | None = None
    best_cost = float("inf")
    expansions = 0
    for _ in range(simulations):
        node = root
        path = [node]
        while len(node.actions) < horizon:
            state = start if len(node.actions) == 0 else node.states[-1]
            limit = topk
            if progressive:
                limit = min(
                    topk,
                    max(1, int(widening_c * max(1, node.visits) ** widening_alpha)),
                )
            if not node.expanded or len(node.children) < limit:
                proposals, probabilities = propose(state, limit)
                known = {child.actions[-1].detach().cpu().numpy().tobytes()
                         for child in node.children}
                for action, probability in zip(proposals, probabilities):
                    key = action.detach().cpu().numpy().tobytes()
                    if key in known:
                        continue
                    actions = torch.cat([node.actions, action[None]], 0)
                    states = transition(actions)
                    node.children.append(_Node(
                        actions, states, float(probability), parent=node
                    ))
                    expansions += 1
                    known.add(key)
                    if progressive and len(node.children) >= limit:
                        break
                node.expanded = (not progressive) or len(node.children) >= topk
                if not node.children:
                    break
            scale = max(1.0, node.visits ** 0.5)
            node = max(
                node.children,
                key=lambda child: (
                    -child.value
                    + c_puct * child.prior * scale / (1 + child.visits)
                ),
            )
            path.append(node)
            if node.visits == 0:
                break
        # Complete a newly expanded prefix with the proposal policy.  This is
        # the standard MCTS rollout/leaf-evaluation step and guarantees that
        # every simulation supplies a horizon-matched value even when the
        # branching factor is much larger than the simulation budget.
        rollout_actions = node.actions
        rollout_states = node.states
        while len(rollout_actions) < horizon:
            state = start if len(rollout_actions) == 0 else rollout_states[-1]
            proposals, probabilities = propose(state, topk)
            action = proposals[int(probabilities.argmax())]
            rollout_actions = torch.cat([rollout_actions, action[None]], 0)
            rollout_states = transition(rollout_actions)
        cost = float(leaf_cost(rollout_states[-1:])[0])
        for visited in path:
            visited.visits += 1
            visited.value_sum += cost
        if cost < best_cost:
            best_actions, best_states, best_cost = (
                rollout_actions.clone(), rollout_states.clone(), cost
            )
    if best_actions is None or best_states is None:
        raise RuntimeError("PUCT did not evaluate a complete rollout")
    return TreeResult(best_actions, best_states, best_cost, {
        "tree_expansions": float(expansions),
        "tree_simulations": float(simulations),
        "root_visits": float(root.visits),
    })
