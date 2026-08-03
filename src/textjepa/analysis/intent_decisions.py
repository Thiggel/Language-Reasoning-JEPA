"""Metrics for aligned intent-policy and GAR geometry audits.

The functions in this module are deliberately model-agnostic.  Audit scripts
emit one row per state/candidate and this module summarizes the ranking signal
without granting a model access to symbolic labels at inference time.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def rank_metrics(costs: Iterable[float], positive: Iterable[bool]) -> dict:
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
    return {
        "top1": float(top1),
        "reciprocal_rank": 1.0 / (first_positive + 1),
        "pairwise_accuracy": pairwise,
        "margin": margin,
        "n_candidates": int(cost.size),
        "n_positive": int(label.sum()),
    }


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
        "margin_mean": mean("margin", competitive),
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
