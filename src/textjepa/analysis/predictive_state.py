"""Representation, rollout, and goal-geometry metrics."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _flatten_states(states: torch.Tensor) -> torch.Tensor:
    if states.ndim < 2:
        raise ValueError("states require a feature dimension")
    return states.reshape(-1, states.shape[-1]).float()


def covariance_eigenvalues(
    states: torch.Tensor, *, maximum_samples: int = 4096
) -> torch.Tensor:
    values = _flatten_states(states)
    if len(values) > maximum_samples:
        indices = torch.linspace(
            0, len(values) - 1, maximum_samples, device=values.device
        ).long()
        values = values[indices]
    centered = values - values.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    return torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)


def effective_rank(eigenvalues: torch.Tensor) -> float:
    total = eigenvalues.sum()
    if float(total) <= 0:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def mean_pairwise_cosine(
    states: torch.Tensor, *, pairs: int = 4096, seed: int = 0
) -> float:
    values = torch.nn.functional.normalize(_flatten_states(states), dim=-1)
    if len(values) < 2:
        return 1.0
    generator = torch.Generator(device=values.device).manual_seed(seed)
    left = torch.randint(len(values), (pairs,), generator=generator,
                         device=values.device)
    right = torch.randint(len(values) - 1, (pairs,), generator=generator,
                          device=values.device)
    right = right + (right >= left)
    return float((values[left] * values[right]).sum(-1).mean())


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = _flatten_states(left)
    right = _flatten_states(right)
    if left.shape != right.shape:
        raise ValueError("CKA inputs must have identical shapes")
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    cross = torch.linalg.matrix_norm(left.T @ right).square()
    left_norm = torch.linalg.matrix_norm(left.T @ left)
    right_norm = torch.linalg.matrix_norm(right.T @ right)
    return float(cross / (left_norm * right_norm).clamp_min(1e-30))


def geometry_summary(states: torch.Tensor) -> dict[str, object]:
    values = _flatten_states(states)
    eigenvalues = covariance_eigenvalues(values)
    norms = values.norm(dim=-1)
    return {
        "samples": len(values),
        "width": values.shape[-1],
        "effective_rank": effective_rank(eigenvalues),
        "mean_pairwise_cosine": mean_pairwise_cosine(values),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(unbiased=False)),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "covariance_eigenvalues_top64": [
            float(value) for value in eigenvalues[:64]
        ],
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten().float()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and sorted_values[end] == sorted_values[begin]:
            end += 1
        ranks[order[begin:end]] = 0.5 * (begin + end - 1)
        begin = end
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel() or left.numel() < 2:
        raise ValueError("Spearman inputs must align and contain two values")
    x = _average_ranks(left)
    y = _average_ranks(right)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if float(denominator) == 0:
        return 0.0
    return float((x * y).sum() / denominator)


def matched_action_geometry(
    current_states: torch.Tensor,
    next_states: torch.Tensor,
    action_ids: torch.Tensor,
    *,
    maximum_pairs: int = 8192,
    seed: int = 0,
) -> dict[str, float | int]:
    """Compare current and post-action distances only within token groups."""
    current = _flatten_states(current_states)
    future = _flatten_states(next_states)
    actions = action_ids.reshape(-1)
    if not (len(current) == len(future) == len(actions)):
        raise ValueError("matched-action inputs do not align")
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    left_parts, right_parts = [], []
    for action in actions.unique():
        indices = torch.nonzero(actions == action).flatten()
        if len(indices) < 2:
            continue
        draws = min(len(indices), max(2, maximum_pairs // 128))
        left = indices[torch.randint(
            len(indices), (draws,), generator=generator, device=indices.device
        )]
        offset = torch.randint(
            len(indices) - 1, (draws,), generator=generator,
            device=indices.device,
        )
        right_positions = offset + (offset >= torch.searchsorted(indices, left))
        right = indices[right_positions]
        left_parts.append(left)
        right_parts.append(right)
        if sum(len(part) for part in left_parts) >= maximum_pairs:
            break
    if not left_parts:
        return {"pairs": 0, "spearman": float("nan")}
    left = torch.cat(left_parts)[:maximum_pairs]
    right = torch.cat(right_parts)[:maximum_pairs]
    current_distance = (current[left] - current[right]).norm(dim=-1)
    future_distance = (future[left] - future[right]).norm(dim=-1)
    return {
        "pairs": len(left),
        "spearman": spearman(current_distance, future_distance),
    }


def progress_metrics(
    distances: Iterable[torch.Tensor],
    *,
    offsets: tuple[int, ...] = (1, 2, 4),
) -> dict[str, object]:
    correlations = []
    monotonic = {offset: [] for offset in offsets}
    for trajectory in distances:
        trajectory = trajectory.flatten().float()
        if len(trajectory) < 2:
            continue
        progress = torch.linspace(0, 1, len(trajectory))
        correlations.append(spearman(-trajectory, progress))
        for offset in offsets:
            if len(trajectory) > offset:
                monotonic[offset].append(float(
                    (trajectory[:-offset] > trajectory[offset:]).float().mean()
                ))
    return {
        "trajectories": len(correlations),
        "progress_spearman_mean": (
            sum(correlations) / len(correlations) if correlations
            else float("nan")
        ),
        "progress_spearman_values": correlations,
        "pairwise_monotonicity": {
            str(offset): (
                sum(values) / len(values) if values else float("nan")
            ) for offset, values in monotonic.items()
        },
        "oracle_terminal_state": True,
    }


def candidate_ranking_accuracy(
    correct_distances: torch.Tensor,
    incorrect_distances: torch.Tensor,
) -> float:
    if correct_distances.shape != incorrect_distances.shape:
        raise ValueError("candidate pairs must align")
    wins = (correct_distances < incorrect_distances).float()
    ties = (correct_distances == incorrect_distances).float()
    return float((wins + 0.5 * ties).mean())


def distance_bellman_residual(
    current_distance: torch.Tensor,
    candidate_next_distances: torch.Tensor,
) -> torch.Tensor:
    if candidate_next_distances.shape[:-1] != current_distance.shape:
        raise ValueError("candidate distances must end in an action axis")
    return (current_distance - (1.0 + candidate_next_distances.min(-1).values)).abs()
