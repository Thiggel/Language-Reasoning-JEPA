"""Metrics for aligned intent-policy and GAR geometry audits.

The functions in this module are deliberately model-agnostic.  Audit scripts
emit one row per state/candidate and this module summarizes the ranking signal
without granting a model access to symbolic labels at inference time.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, assigning equal values equal rank."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def rank_metrics(
    costs: Iterable[float],
    positive: Iterable[bool],
    target_costs: Iterable[float] | None = None,
) -> dict:
    """Summarize a lower-is-better candidate ranking.

    Multiple actions may be optimal/necessary.  Top-1 is correct if any
    minimum-cost action is positive, reciprocal rank uses the first positive,
    and pairwise accuracy compares every positive-negative pair.  Exact ties
    receive half credit in the pairwise statistic.
    """

    cost = np.asarray(list(costs), dtype=np.float64)
    label = np.asarray(list(positive), dtype=bool)
    if cost.ndim != 1 or label.shape != cost.shape or cost.size == 0:
        raise ValueError("costs and positive must be non-empty aligned vectors")
    if not label.any():
        raise ValueError("at least one candidate must be positive")
    finite = np.isfinite(cost)
    if not finite.all():
        raise ValueError("candidate costs must be finite")

    order = np.argsort(cost, kind="stable")
    first_positive = int(np.flatnonzero(label[order])[0])
    minimum = cost.min()
    top_ties = cost == minimum
    top1 = bool((label & top_ties).any())
    positive_cost = cost[label]
    negative_cost = cost[~label]
    if negative_cost.size:
        differences = negative_cost[None, :] - positive_cost[:, None]
        pairwise = float(
            (differences > 0).mean() + 0.5 * (differences == 0).mean()
        )
        margin = float(negative_cost.min() - positive_cost.min())
    else:
        pairwise = 1.0
        margin = float("nan")
    metrics = {
        "top1": float(top1),
        "reciprocal_rank": 1.0 / (first_positive + 1),
        "pairwise_accuracy": pairwise,
        "margin": margin,
        "n_candidates": int(cost.size),
        "n_positive": int(label.sum()),
    }
    if target_costs is None:
        target = (~label).astype(np.float64)
    else:
        target = np.asarray(list(target_costs), dtype=np.float64)
        if target.shape != cost.shape or not np.isfinite(target).all():
            raise ValueError("target_costs must be finite and align with costs")
    optimal = target == target.min()
    selected = int(np.argmin(cost))
    metrics["mean_regret"] = float(target[selected] - target.min())
    metrics["optimal_top1"] = float((optimal & top_ties).any())
    metrics["optimal_margin"] = (
        float(cost[~optimal].min() - cost[optimal].min())
        if (~optimal).any() else float("nan")
    )
    target_diff = target[:, None] - target[None, :]
    score_diff = cost[:, None] - cost[None, :]
    informative = target_diff < 0
    metrics["exact_ordering_accuracy"] = (
        float(
            (score_diff[informative] < 0).mean()
            + 0.5 * (score_diff[informative] == 0).mean()
        )
        if informative.any() else float("nan")
    )
    target_rank = _average_ranks(target)
    score_rank = _average_ranks(cost)
    metrics["spearman"] = (
        float(np.corrcoef(target_rank, score_rank)[0, 1])
        if target_rank.std() > 0 and score_rank.std() > 0
        else float("nan")
    )
    return metrics


def aggregate_rank_metrics(rows: Iterable[dict]) -> dict:
    """Macro-average decision metrics, retaining competitive-state counts."""

    values = list(rows)
    if not values:
        return {"decisions": 0, "competitive_decisions": 0}
    competitive = [row for row in values if row["n_candidates"] > row["n_positive"]]

    def mean(key: str, selected: list[dict]) -> float | None:
        finite = [float(row[key]) for row in selected if np.isfinite(row[key])]
        return float(np.mean(finite)) if finite else None

    return {
        "decisions": len(values),
        "competitive_decisions": len(competitive),
        "top1": mean("top1", values),
        "competitive_top1": mean("top1", competitive),
        "reciprocal_rank": mean("reciprocal_rank", values),
        "pairwise_accuracy": mean("pairwise_accuracy", competitive),
        "exact_ordering_accuracy": mean("exact_ordering_accuracy", competitive),
        "optimal_top1": mean("optimal_top1", values),
        "mean_regret": mean("mean_regret", values),
        "spearman": mean("spearman", competitive),
        "margin_mean": mean("margin", competitive),
        "optimal_margin_mean": mean("optimal_margin", competitive),
        "margin_median": (
            float(np.median([row["margin"] for row in competitive]))
            if competitive else None
        ),
        "near_tie_fraction": {
            str(threshold): (
                float(np.mean([
                    abs(row["margin"]) < threshold for row in competitive
                ]))
                if competitive else None
            )
            for threshold in (0.005, 0.01, 0.02, 0.05)
        },
    }


def spearman_tied(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Spearman correlation with deterministic average ranks for ties."""

    left_array = np.asarray(list(left), dtype=np.float64)
    right_array = np.asarray(list(right), dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("Spearman inputs must be aligned vectors")
    left_rank, right_rank = _average_ranks(left_array), _average_ranks(right_array)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def summarize_depth_decision(
    sequences: list[list[int | None]],
    sequence_costs: Iterable[float],
    endpoint_remaining: Iterable[float],
    root_exact_cost: dict[int, int],
    root_is_necessary: dict[int, bool],
    one_step_cost: dict[int, float],
) -> tuple[dict, list[dict]]:
    """Summarize sequence selection and root-action ordering for one state."""

    costs = np.asarray(list(sequence_costs), dtype=np.float64)
    remaining = np.asarray(list(endpoint_remaining), dtype=np.float64)
    if len(sequences) != len(costs) or costs.shape != remaining.shape:
        raise ValueError("sequences, costs, and endpoint labels must align")
    roots = sorted(root_exact_cost)
    rows = []
    deployed_root_costs = []
    root_only_costs = []
    oracle_endpoint_costs = []
    labels = []
    exact = []
    for root in roots:
        indices = np.asarray(
            [index for index, sequence in enumerate(sequences) if sequence[0] == root]
        )
        selected = int(indices[np.argmin(costs[indices])])
        deployed = float(costs[selected])
        endpoint = float(remaining[selected])
        best_endpoint = float(remaining[indices].min())
        deployed_root_costs.append(deployed)
        root_only_costs.append(float(one_step_cost[root]))
        oracle_endpoint_costs.append(best_endpoint)
        labels.append(root_is_necessary[root])
        exact.append(float(root_exact_cost[root]))
        rows.append({
            "root": int(root),
            "rollouts": int(len(indices)),
            "deployed_root_score": deployed,
            "one_step_root_score": float(one_step_cost[root]),
            "selected_endpoint_remaining": endpoint,
            "best_endpoint_remaining": best_endpoint,
            "within_root_endpoint_regret": endpoint - best_endpoint,
            "score_min": float(costs[indices].min()),
            "score_mean": float(costs[indices].mean()),
            "score_std": float(costs[indices].std()),
            "selection_optimism": float(costs[indices].mean() - costs[indices].min()),
            "necessary": bool(root_is_necessary[root]),
            "root_exact_cost": int(root_exact_cost[root]),
        })
    metrics = {
        "deployed_terminal_score": rank_metrics(deployed_root_costs, labels, exact),
        "one_step_root_score": rank_metrics(root_only_costs, labels, exact),
        "oracle_endpoint_score": rank_metrics(oracle_endpoint_costs, labels, exact),
        "sequence_score_endpoint_spearman": spearman_tied(costs, remaining),
        "selected_sequence_endpoint_regret": float(
            remaining[int(np.argmin(costs))] - remaining.min()
        ),
        "selection_optimism": float(costs.mean() - costs.min()),
    }
    return metrics, rows


def transition_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Measure drift and within-state candidate retrieval.

    ``groups`` identifies the decision state for each candidate.  Retrieval is
    correct when a predicted next state is nearest to its own true next state
    among the true candidates available at that same state.
    """

    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must be aligned matrices")
    if len(groups) != len(predicted):
        raise ValueError("groups must align with candidate rows")
    if not len(predicted):
        return {"candidates": 0}

    def layer_norm(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + 1e-5
        )

    p = layer_norm(predicted)
    t = layer_norm(target)
    l1 = np.abs(p - t).mean(-1)
    p_delta = p - p.mean(-1, keepdims=True)
    t_delta = t - t.mean(-1, keepdims=True)
    cosine = (p_delta * t_delta).sum(-1) / (
        np.linalg.norm(p_delta, axis=-1) * np.linalg.norm(t_delta, axis=-1)
        + 1e-12
    )
    retrieval = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        distances = np.abs(
            p[indices, None, :] - t[indices][None, :, :]
        ).mean(-1)
        retrieval.extend((distances.argmin(-1) == np.arange(len(indices))).tolist())
    return {
        "candidates": int(len(predicted)),
        "layernorm_l1_mean": float(l1.mean()),
        "layernorm_l1_median": float(np.median(l1)),
        "state_cosine_mean": float(cosine.mean()),
        "within_state_retrieval": float(np.mean(retrieval)),
    }


def geometry_features(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Goal-relative features for an information-matched linear probe."""

    state = np.asarray(state, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if state.shape != goal.shape or state.ndim != 2:
        raise ValueError("state and goal must be aligned matrices")
    return np.concatenate(
        [state, goal, state - goal, np.abs(state - goal), state * goal], axis=1
    )
