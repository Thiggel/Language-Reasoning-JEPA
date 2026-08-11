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


def _aggregate(
    results: list[EpisodeResult], max_slack: int | None = None
) -> dict[str, float]:
    n = len(results)
    metrics = {
        "success": sum(r.solved for r in results) / n,
        "mean_steps": sum(r.steps for r in results) / n,
        "mean_necessary": sum(r.n_necessary for r in results) / n,
        "distractor_rate": sum(r.n_distractor for r in results)
        / max(sum(r.steps for r in results), 1),
        "invalid_action_rate": sum(r.n_invalid for r in results)
        / max(sum(r.steps for r in results), 1),
    }
    ranks = [r.prior_rank for r in results if r.prior_rank is not None]
    if ranks:
        # Learned-catalogue diagnostic: mean rank of the ground-truth next
        # action under the learned prior (1 = the prior's top choice).
        metrics["mean_prior_rank"] = sum(ranks) / len(ranks)
    recalls = [
        r.proposal_recall for r in results if r.proposal_recall is not None
    ]
    if recalls:
        # generator_cycle diagnostic: fraction of steps at which at least one
        # truly feasible action appeared among the parsed generator proposals
        # (measurement only; proposals and scoring never see the oracle).
        metrics["proposal_recall"] = sum(recalls) / len(recalls)
    parse_rates = [
        r.parse_rate for r in results if r.parse_rate is not None
    ]
    if parse_rates:
        metrics["proposal_parse_rate"] = sum(parse_rates) / len(parse_rates)
    if max_slack is not None:
        # The policy never reads the remaining budget, so one run at
        # slack=max_slack yields every smaller-slack success rate exactly:
        # an episode solved with excess e behaves identically under any
        # budget >= necessary + e.
        excess = [
            r.steps - r.n_necessary if r.solved else None for r in results
        ]
        metrics["success_by_slack"] = {
            str(k): sum(e is not None and e <= k for e in excess) / n
            for k in range(max_slack + 1)
        }
        metrics["excess_steps"] = [
            e if e is not None else -1 for e in excess
        ]
    return metrics


def evaluate_planning(
    planner: LatentPlanner,
    dataset,
    n_episodes: int,
    slack: int = 0,
    seed: int = 0,
    slack_curve: bool = False,
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
    max_slack = slack if slack_curve else None
    planned_metrics = _aggregate(planned, max_slack)
    if hasattr(planner, "n_macro_decisions"):
        total = planner.n_macro_decisions + planner.n_flat_decisions
        planned_metrics.update({
            "macro_decision_rate": planner.n_macro_decisions / max(total, 1),
            "macro_decisions": float(planner.n_macro_decisions),
            "flat_decisions": float(planner.n_flat_decisions),
        })
    return {
        planner_name: planned_metrics,
        "random_policy": _aggregate(rand_, max_slack),
        "first_feasible_policy": _aggregate(first_, max_slack),
        "oracle": _aggregate(oracle, max_slack),
    }
