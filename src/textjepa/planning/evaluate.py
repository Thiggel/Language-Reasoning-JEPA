"""Planning evaluation: latent planner vs symbolic baselines."""

from __future__ import annotations

import random

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import Problem
from textjepa.planning.search import EpisodeResult, LatentPlanner


def _environment(problem):
    if hasattr(problem, "params") and hasattr(problem, "necessary"):
        from textjepa.data.faithful import FaithfulEnv

        return FaithfulEnv(problem)
    return SymbolicEnv(problem)


def _necessary(problem) -> set:
    return (
        set(problem.necessary)
        if hasattr(problem, "necessary")
        else set(problem.query_ancestors)
    )


def random_policy_episode(problem: Problem, slack: int, rng: random.Random) -> EpisodeResult:
    env = _environment(problem)
    necessary = _necessary(problem)
    budget = len(necessary) + slack
    n_distractor = 0
    while not env.solved and len(env.resolved) < budget:
        a = rng.choice(env.feasible_actions())
        n_distractor += int(a not in necessary)
        env.step(a)
    return EpisodeResult(env.solved, len(env.resolved), len(necessary), n_distractor)


def first_feasible_episode(problem: Problem, slack: int) -> EpisodeResult:
    """Ordering-control diagnostic for candidate enumeration artifacts."""
    env = _environment(problem)
    necessary = _necessary(problem)
    budget = len(necessary) + slack
    n_distractor = 0
    while not env.solved and len(env.resolved) < budget:
        action = env.feasible_actions()[0]
        n_distractor += int(action not in necessary)
        env.step(action)
    return EpisodeResult(
        env.solved, len(env.resolved), len(necessary), n_distractor
    )


def oracle_episode(problem: Problem, rng: random.Random) -> EpisodeResult:
    env = _environment(problem)
    necessary_set = _necessary(problem)
    while not env.solved:
        necessary = [a for a in env.feasible_actions() if a in necessary_set]
        env.step(rng.choice(necessary))
    return EpisodeResult(True, len(env.resolved), len(necessary_set), 0)


def _aggregate(results: list[EpisodeResult]) -> dict[str, float]:
    n = len(results)
    return {
        "success": sum(r.solved for r in results) / n,
        "mean_steps": sum(r.steps for r in results) / n,
        "mean_necessary": sum(r.n_necessary for r in results) / n,
        "distractor_rate": sum(r.n_distractor for r in results)
        / max(sum(r.steps for r in results), 1),
    }


def evaluate_planning(
    planner: LatentPlanner,
    dataset,
    n_episodes: int,
    slack: int = 0,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    planned, rand_, first_, oracle = [], [], [], []
    for i in range(n_episodes):
        problem, _ = dataset.problem(i)
        planned.append(planner.plan_episode(problem, slack=slack, seed=seed + i))
        rand_.append(random_policy_episode(problem, slack, rng))
        first_.append(first_feasible_episode(problem, slack))
        oracle.append(oracle_episode(problem, rng))
    planner_name = (
        "latent_planner" if planner.energy == "value" else f"latent_planner_{planner.energy}"
    )
    planned_metrics = _aggregate(planned)
    if hasattr(planner, "n_macro_decisions"):
        total = planner.n_macro_decisions + planner.n_flat_decisions
        planned_metrics.update({
            "macro_decision_rate": planner.n_macro_decisions / max(total, 1),
            "macro_decisions": float(planner.n_macro_decisions),
            "flat_decisions": float(planner.n_flat_decisions),
        })
    return {
        planner_name: planned_metrics,
        "random_policy": _aggregate(rand_),
        "first_feasible_policy": _aggregate(first_),
        "oracle": _aggregate(oracle),
    }
