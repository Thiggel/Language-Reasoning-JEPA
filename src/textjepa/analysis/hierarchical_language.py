"""Decision-grade diagnostics for hierarchical language planning."""

from __future__ import annotations

import torch


HIERARCHICAL_PROBE_TARGETS = (
    "canonical_symbolic_state", "active_equations", "variable_values",
    "next_valid_operation", "proposed_operation_validity",
    "proposed_resulting_state", "terminality", "remaining_shortest_path",
    "future_state_h1", "future_state_h2", "future_state_h4",
    "paraphrase_identity", "template_identity", "previous_token",
    "sentence_length", "frozen_lm_next_token_logits",
)


@torch.no_grad()
def representation_statistics(states: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """Collapse, redundancy, scale, and spectrum diagnostics for ``[N,D]``."""
    if states.ndim != 2 or len(states) < 2:
        raise ValueError("states must be a [samples, dimensions] matrix")
    value = states.float()
    mean = value.mean(0)
    centered = value - mean
    covariance = centered.T @ centered / (len(value) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    tolerance = torch.finfo(eigenvalues.dtype).eps * max(
        1.0, float(eigenvalues[0])
    )
    condition = (
        float(eigenvalues[0] / eigenvalues[-1])
        if bool(eigenvalues[-1] > tolerance)
        else float("inf")
    )
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    nonzero = probability[probability > 0]
    entropic_rank = float(torch.exp(-(nonzero * nonzero.log()).sum()))
    participation = float(
        eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
    )
    return {
        "mean": mean,
        "std": value.std(0),
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "condition_number": condition,
        "entropic_effective_rank": entropic_rank,
        "participation_ratio": participation,
        "mean_norm": float(value.norm(dim=-1).mean()),
    }


@torch.no_grad()
def proposal_coverage(
    valid_complete_solution: torch.Tensor,
    sample_counts: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128),
) -> dict[str, float]:
    """Greedy and oracle@N coverage from ordered frozen-LM samples."""
    if valid_complete_solution.ndim != 2 or valid_complete_solution.dtype != torch.bool:
        raise ValueError("coverage input must be boolean [problems, samples]")
    result = {"greedy": float(valid_complete_solution[:, 0].float().mean())}
    for count in sample_counts:
        if count < 1 or count > valid_complete_solution.shape[1]:
            continue
        result[f"oracle@{count}"] = float(
            valid_complete_solution[:, :count].any(-1).float().mean()
        )
    return result


@torch.no_grad()
def value_ranking_diagnostics(
    teacher_cost: torch.Tensor,
    predicted_cost: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Pairwise accuracy, top-one regret, and NDCG for within-root rankings."""
    if teacher_cost.shape != predicted_cost.shape or mask.shape != teacher_cost.shape:
        raise ValueError("ranking tensors must align")
    pair_correct, pair_count, regrets, ndcgs = 0.0, 0, [], []
    spearman_values, kendall_values = [], []
    for target, prediction, valid in zip(teacher_cost, predicted_cost, mask):
        target, prediction = target[valid], prediction[valid]
        if len(target) < 1:
            continue
        selected = int(prediction.argmin())
        regrets.append(float(target[selected] - target.min()))
        if len(target) > 1:
            left, right = torch.triu_indices(len(target), len(target), offset=1)
            product = (target[left] - target[right]) * (
                prediction[left] - prediction[right]
            )
            pair_correct += float((product > 0).sum())
            pair_count += len(left)
            kendall_values.append(float(
                (product.sign().sum()) / max(len(product), 1)
            ))
            target_rank = target.argsort().argsort().float()
            predicted_rank = prediction.argsort().argsort().float()
            target_centered = target_rank - target_rank.mean()
            predicted_centered = predicted_rank - predicted_rank.mean()
            spearman_values.append(float(
                (target_centered * predicted_centered).sum()
                / (
                    target_centered.square().sum().sqrt()
                    * predicted_centered.square().sum().sqrt()
                ).clamp_min(1e-12)
            ))
        relevance = torch.exp(-(target - target.min()))
        order = prediction.argsort()
        discounts = 1 / torch.log2(
            torch.arange(len(target), device=target.device).float() + 2
        )
        dcg = (relevance[order] * discounts).sum()
        ideal = (relevance.sort(descending=True).values * discounts).sum()
        ndcgs.append(float(dcg / ideal.clamp_min(1e-12)))
    if not regrets:
        raise ValueError("no valid ranking roots")
    return {
        "pairwise_accuracy": pair_correct / max(pair_count, 1),
        "top_one_regret": sum(regrets) / len(regrets),
        "ndcg": sum(ndcgs) / len(ndcgs),
        "spearman": (
            sum(spearman_values) / len(spearman_values)
            if spearman_values else float("nan")
        ),
        "kendall": (
            sum(kendall_values) / len(kendall_values)
            if kendall_values else float("nan")
        ),
    }


@torch.no_grad()
def requested_achieved_displacement(
    requested: torch.Tensor, achieved: torch.Tensor
) -> torch.Tensor:
    if requested.shape != achieved.shape:
        raise ValueError("requested and achieved waypoints must align")
    return requested - achieved


@torch.no_grad()
def path_geometry(states: torch.Tensor) -> dict[str, torch.Tensor]:
    """Path-to-chord ratios and turning cosines for latent trajectories."""
    if states.ndim != 3 or states.shape[1] < 2:
        raise ValueError("states must be [trajectories, time, dimensions]")
    delta = states[:, 1:] - states[:, :-1]
    path = delta.norm(dim=-1).sum(-1)
    chord = (states[:, -1] - states[:, 0]).norm(dim=-1)
    ratio = path / chord.clamp_min(1e-12)
    if delta.shape[1] < 2:
        turning = states.new_empty(states.shape[0], 0)
    else:
        turning = torch.nn.functional.cosine_similarity(
            delta[:, :-1], delta[:, 1:], dim=-1
        )
    return {"path_to_chord": ratio, "turning_cosine": turning}


@torch.no_grad()
def shortcut_error_ratios(
    full_error: torch.Tensor,
    **shortcut_errors: torch.Tensor,
) -> dict[str, float]:
    """Compare identity/action/state/cache shortcuts with the full predictor."""
    if full_error.ndim != 1 or not torch.isfinite(full_error).all():
        raise ValueError("full errors must be a finite vector")
    denominator = full_error.mean().clamp_min(1e-12)
    result = {"full_error": float(full_error.mean())}
    for name, error in shortcut_errors.items():
        if error.shape != full_error.shape or not torch.isfinite(error).all():
            raise ValueError(f"shortcut error {name} is invalid")
        result[f"{name}_error"] = float(error.mean())
        result[f"{name}_to_full"] = float(error.mean() / denominator)
    return result


@torch.no_grad()
def endpoint_grounding_diagnostics(
    predicted_cost: torch.Tensor,
    exact_cost: torch.Tensor,
    achieved_cost: torch.Tensor | None = None,
) -> dict[str, float]:
    """Planner-model gap and optimizer selection regret."""
    if predicted_cost.ndim != 1 or exact_cost.shape != predicted_cost.shape:
        raise ValueError("endpoint costs must be aligned vectors")
    if not torch.isfinite(predicted_cost).all() or not torch.isfinite(exact_cost).all():
        raise ValueError("endpoint costs must be finite")
    selected = int(predicted_cost.argmin())
    result = {
        "mean_absolute_model_gap": float(
            (predicted_cost - exact_cost).abs().mean()
        ),
        "selected_exact_cost": float(exact_cost[selected]),
        "best_exact_cost": float(exact_cost.min()),
        "exact_selection_regret": float(
            exact_cost[selected] - exact_cost.min()
        ),
    }
    if achieved_cost is not None:
        if achieved_cost.shape != predicted_cost.shape or not torch.isfinite(
            achieved_cost
        ).all():
            raise ValueError("achieved costs must align and be finite")
        result.update({
            "selected_achieved_cost": float(achieved_cost[selected]),
            "best_achieved_cost": float(achieved_cost.min()),
            "achieved_selection_regret": float(
                achieved_cost[selected] - achieved_cost.min()
            ),
        })
    return result


@torch.no_grad()
def fiber_diagnostics(
    fine_states: torch.Tensor,
    coarse_states: torch.Tensor,
    canonical_state_ids: torch.Tensor,
    wording_ids: torch.Tensor,
    *,
    radius: float,
) -> dict[str, float]:
    """Measure symbolic agreement and textual diversity in coarse fibers."""
    if radius <= 0:
        raise ValueError("fiber radius must be positive")
    count = len(coarse_states)
    if fine_states.shape[0] != count or canonical_state_ids.shape != (count,):
        raise ValueError("fiber tensors do not align")
    distance = torch.cdist(coarse_states.float(), coarse_states.float())
    pairs = torch.triu(distance <= radius, diagonal=1)
    left, right = pairs.nonzero(as_tuple=True)
    if len(left) == 0:
        return {
            "fiber_pairs": 0.0,
            "canonical_agreement": float("nan"),
            "wording_diversity": float("nan"),
            "fine_diameter": float("nan"),
        }
    return {
        "fiber_pairs": float(len(left)),
        "canonical_agreement": float(
            (canonical_state_ids[left] == canonical_state_ids[right]).float().mean()
        ),
        "wording_diversity": float(
            (wording_ids[left] != wording_ids[right]).float().mean()
        ),
        "fine_diameter": float(
            (fine_states[left] - fine_states[right]).norm(dim=-1).max()
        ),
    }


def linear_probe_diagnostics(
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    test_states: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    classification: bool,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Fit an explicit held-out linear probe with ridge regularization."""
    if ridge < 0 or train_states.ndim != 2 or test_states.ndim != 2:
        raise ValueError("invalid probe configuration")
    if train_states.shape[1] != test_states.shape[1] or (
        len(train_states) != len(train_targets)
    ) or len(test_states) != len(test_targets):
        raise ValueError("probe tensors do not align")
    x_train = torch.cat([
        train_states.float(),
        torch.ones(len(train_states), 1, device=train_states.device),
    ], -1)
    x_test = torch.cat([
        test_states.float(),
        torch.ones(len(test_states), 1, device=test_states.device),
    ], -1)
    if classification:
        # The label vocabulary is part of the fitted probe and therefore may
        # only be inferred from the training partition.
        classes = torch.unique(train_targets.reshape(-1)).sort().values
        if len(classes) < 2:
            raise ValueError("classification probe needs at least two classes")
        encoded = (
            train_targets.reshape(-1, 1) == classes.reshape(1, -1)
        ).float()
        weight = torch.linalg.solve(
            x_train.T @ x_train
            + ridge * torch.eye(x_train.shape[1], device=x_train.device),
            x_train.T @ encoded,
        )
        predicted = classes[(x_test @ weight).argmax(-1)]
        return {"accuracy": float(
            (predicted == test_targets.reshape(-1)).float().mean()
        )}
    target = train_targets.float()
    if target.ndim == 1:
        target = target[:, None]
    weight = torch.linalg.solve(
        x_train.T @ x_train
        + ridge * torch.eye(x_train.shape[1], device=x_train.device),
        x_train.T @ target,
    )
    prediction = x_test @ weight
    expected = test_targets.float().reshape_as(prediction)
    residual = (prediction - expected).square().mean()
    baseline = (expected - expected.mean(0)).square().mean().clamp_min(1e-12)
    return {
        "mse": float(residual),
        "r2": float(1 - residual / baseline),
    }


@torch.no_grad()
def cache_window_sweep(
    evaluate_window,
    windows: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32),
) -> dict[str, float]:
    """Run the same endpoint diagnostic under explicit cache windows."""
    if not windows or any(window < 0 for window in windows):
        raise ValueError("cache windows must be nonnegative")
    result = {}
    for window in windows:
        error = torch.as_tensor(evaluate_window(window)).float()
        if error.numel() < 1 or not torch.isfinite(error).all():
            raise ValueError("cache-window evaluator returned invalid errors")
        result[f"window_{window}_error"] = float(error.mean())
    return result


@torch.no_grad()
def counterfactual_generation_diagnostics(
    log_probability: torch.Tensor,
    lengths: torch.Tensor,
    source_ids: torch.Tensor,
    valid_step: torch.Tensor,
    symbolic_delta: torch.Tensor,
    endpoint: torch.Tensor,
    token_ids: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Summarize candidate plausibility/diversity/progress by source."""
    count = len(log_probability)
    aligned = (lengths, source_ids, valid_step, symbolic_delta)
    if any(value.shape != (count,) for value in aligned) or (
        endpoint.shape[0] != count
    ):
        raise ValueError("counterfactual diagnostic tensors do not align")
    if not torch.isfinite(log_probability).all() or not torch.isfinite(
        endpoint
    ).all():
        raise ValueError("counterfactual diagnostics must be finite")
    result = {}
    for source in torch.unique(source_ids):
        mask = source_ids == source
        values = endpoint[mask].float()
        diversity = (
            torch.pdist(values).mean() if len(values) > 1
            else values.new_zeros(())
        )
        result[str(int(source))] = {
            "count": float(mask.sum()),
            "mean_log_probability": float(log_probability[mask].mean()),
            "mean_length": float(lengths[mask].float().mean()),
            "valid_step_rate": float(valid_step[mask].float().mean()),
            "improving_rate": float(
                (symbolic_delta[mask] < 0).float().mean()
            ),
            "unchanged_rate": float(
                (symbolic_delta[mask] == 0).float().mean()
            ),
            "worsening_rate": float(
                (symbolic_delta[mask] > 0).float().mean()
            ),
            "latent_endpoint_diversity": float(diversity),
        }
        if token_ids is not None:
            if token_ids.shape[0] != count or token_mask is None or (
                token_mask.shape != token_ids.shape
            ):
                raise ValueError("counterfactual token diagnostics do not align")
            rows = [
                tuple(token_ids[index][token_mask[index]].tolist())
                for index in mask.nonzero().flatten().tolist()
            ]
            unique = len(set(rows))
            result[str(int(source))].update({
                "unique_continuation_rate": unique / max(len(rows), 1),
                "duplicate_rate": 1 - unique / max(len(rows), 1),
            })
    return result


@torch.no_grad()
def macro_action_diagnostics(
    actions: torch.Tensor,
    posterior_prior_kl: torch.Tensor,
    prior_nll: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Report macro utilization, effective rank, and prior support."""
    if actions.ndim != 2 or posterior_prior_kl.shape != (len(actions),) or (
        prior_nll.shape != (len(actions),)
    ):
        raise ValueError("macro diagnostic tensors do not align")
    spectrum = torch.linalg.eigvalsh(torch.cov(actions.float().T)).clamp_min(0)
    participation = spectrum.sum().square() / spectrum.square().sum().clamp_min(
        epsilon
    )
    return {
        "posterior_prior_kl": float(posterior_prior_kl.mean()),
        "prior_nll": float(prior_nll.mean()),
        "action_effective_rank": float(participation),
        "mean_action_norm": float(actions.norm(dim=-1).mean()),
    }


@torch.no_grad()
def predictor_shortcut_diagnostics(
    predictor,
    states: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    metric,
) -> dict[str, float]:
    """Execute identity/action/state/shuffle/cache shortcut controls."""
    if states.shape != targets.shape or states.shape[:-1] != actions.shape[:-1]:
        raise ValueError("predictor diagnostic tensors do not align")
    if valid.shape != states.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("predictor diagnostic mask is invalid")

    def error(prediction):
        return metric(prediction, targets)[valid]

    full = predictor(states, actions, valid)
    zero_state = torch.zeros_like(states)
    zero_action = torch.zeros_like(actions)
    permutation = torch.arange(len(states) - 1, -1, -1, device=states.device)
    controls = {
        "identity": error(states),
        "action_only": error(predictor(zero_state, actions, valid)),
        "state_only": error(predictor(states, zero_action, valid)),
        "state_shuffled": error(
            predictor(states[permutation], actions, valid)
        ),
        "cache_only": error(predictor(zero_state, zero_action, valid)),
    }
    return shortcut_error_ratios(error(full), **controls)
