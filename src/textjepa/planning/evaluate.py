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
        # Open-ended-interface diagnostic (generator_cycle and cem_cycle
        # report identical fields): fraction of steps at which at least one
        # truly feasible action appeared among the parsed proposals
        # (measurement only; proposals and scoring never see the oracle).
        metrics["proposal_recall"] = sum(recalls) / len(recalls)
    parse_rates = [
        r.parse_rate for r in results if r.parse_rate is not None
    ]
    if parse_rates:
        metrics["proposal_parse_rate"] = sum(parse_rates) / len(parse_rates)
    counts = [
        r.proposal_counts for r in results if r.proposal_counts is not None
    ]
    if counts:
        # Raw per-step proposal counters, averaged over episodes: how many
        # phrases were proposed, how many parsed, how many distinct actions
        # survived, and how many the search finally received.
        for key in counts[0]:
            metrics[f"proposal_{key}"] = (
                sum(c[key] for c in counts) / len(counts)
            )
        # Fraction of episodes that stalled because a step produced no usable
        # candidate at all (the planner never falls back to a menu).
        metrics["no_proposal_episode_rate"] = (
            sum(r.n_no_proposal > 0 for r in results) / n
        )
    return metrics


def slack_curve_metrics(
    results: list[EpisodeResult], slack: int
) -> dict[str, object]:
    """Score ONE generous-budget run at every slack from 0 to ``slack``.

    The policy never reads its budget, so a single episode run at the largest
    slack contains the answer for every smaller slack: the episode counts as a
    success at slack ``s`` exactly when it reached the goal within
    ``optimal + s`` executed actions. ``excess_steps`` reports, per episode,
    how many actions beyond the reference plan length were needed (``None``
    when the episode never solved).
    """
    n = max(len(results), 1)
    # ``solved_at`` is the step index at which the goal was first reached.
    # Policies that stop exactly at the goal (the reference baselines) leave
    # it unset, in which case the executed-step count is the same quantity.
    excess = [
        None if not r.solved else
        (r.steps if r.solved_at is None else r.solved_at) - r.n_necessary
        for r in results
    ]
    return {
        "success_by_slack": {
            str(s): sum(1 for e in excess if e is not None and e <= s) / n
            for s in range(int(slack) + 1)
        },
        "excess_steps": excess,
    }


def aggregate_episodes(
    results: list[EpisodeResult],
    *,
    slack_curve: bool = False,
    slack: int = 0,
) -> dict[str, object]:
    """Planning-metrics JSON shape shared by every domain's evaluator."""
    metrics: dict[str, object] = dict(_aggregate(results))
    if slack_curve:
        metrics.update(slack_curve_metrics(results, slack))
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
    def agg(results: list[EpisodeResult]) -> dict[str, object]:
        return aggregate_episodes(
            results, slack_curve=slack_curve, slack=slack
        )

    planned_metrics = agg(planned)
    if hasattr(planner, "n_macro_decisions"):
        total = planner.n_macro_decisions + planner.n_flat_decisions
        planned_metrics.update({
            "macro_decision_rate": planner.n_macro_decisions / max(total, 1),
            "macro_decisions": float(planner.n_macro_decisions),
            "flat_decisions": float(planner.n_flat_decisions),
        })
    return {
        planner_name: planned_metrics,
        "random_policy": agg(rand_),
        "first_feasible_policy": agg(first_),
        "oracle": agg(oracle),
    }
