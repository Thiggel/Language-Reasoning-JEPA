"""Planning evaluation: latent planner vs symbolic baselines."""

from __future__ import annotations

import random

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import Problem
from textjepa.planning.search import EpisodeResult, LatentPlanner


def random_policy_episode(problem: Problem, slack: int, rng: random.Random) -> EpisodeResult:
    env = SymbolicEnv(problem)
    budget = problem.n_necessary_steps + slack
    n_distractor = 0
    while not env.solved and len(env.resolved) < budget:
        a = rng.choice(env.feasible_actions())
        n_distractor += int(a not in problem.query_ancestors)
        env.step(a)
    return EpisodeResult(env.solved, len(env.resolved), problem.n_necessary_steps, n_distractor)


def oracle_episode(problem: Problem, rng: random.Random) -> EpisodeResult:
    env = SymbolicEnv(problem)
    while not env.solved:
        necessary = [a for a in env.feasible_actions() if a in problem.query_ancestors]
        env.step(rng.choice(necessary))
    return EpisodeResult(True, len(env.resolved), problem.n_necessary_steps, 0)


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
    planned, rand_, oracle = [], [], []
    for i in range(n_episodes):
        problem, _ = dataset.problem(i)
        planned.append(planner.plan_episode(problem, slack=slack, seed=seed + i))
        rand_.append(random_policy_episode(problem, slack, rng))
        oracle.append(oracle_episode(problem, rng))
    planner_name = (
        "latent_planner" if planner.energy == "value" else f"latent_planner_{planner.energy}"
    )
    return {
        planner_name: _aggregate(planned),
        "random_policy": _aggregate(rand_),
        "oracle": _aggregate(oracle),
    }
