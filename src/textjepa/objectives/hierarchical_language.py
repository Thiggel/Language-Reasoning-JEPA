"""Losses and geometry for hierarchical language predictive states."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / (
        weight.sum() * math.prod(value.shape[mask.ndim:])
    ).clamp_min(1)


class EMAShrunkMahalanobis(nn.Module):
    """EMA covariance with isotropic shrinkage and stable linear solves."""

    def __init__(
        self,
        dimension: int,
        momentum: float = 0.99,
        shrinkage: float = 0.1,
        epsilon: float = 1e-4,
        normalized: bool = False,
    ):
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if not 0 <= momentum < 1 or not 0 <= shrinkage <= 1:
            raise ValueError("invalid covariance hyperparameter")
        self.dimension = dimension
        self.momentum = float(momentum)
        self.shrinkage = float(shrinkage)
        self.epsilon = float(epsilon)
        self.normalized = bool(normalized)
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("covariance", torch.eye(dimension))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        flat = states.reshape(-1, self.dimension)
        if mask is not None:
            flat = flat[mask.reshape(-1)]
        if len(flat) < 2:
            return
        mean = flat.mean(0)
        centered = flat - mean
        covariance = centered.T @ centered / (len(flat) - 1)
        weight = 0.0 if int(self.updates) == 0 else self.momentum
        previous_mean = self.mean.clone()
        mean_delta = previous_mean - mean
        mixture_covariance = (
            weight * self.covariance
            + (1 - weight) * covariance
            + weight * (1 - weight) * mean_delta[:, None] @ mean_delta[None, :]
        )
        self.mean.lerp_(mean, 1 - weight)
        self.covariance.copy_(mixture_covariance)
        self.updates.add_(1)

    def matrix(self) -> torch.Tensor:
        scale = self.covariance.trace() / self.dimension
        shrunk = (
            (1 - self.shrinkage) * self.covariance
            + self.shrinkage * scale * torch.eye(
                self.dimension,
                device=self.covariance.device,
                dtype=self.covariance.dtype,
            )
        )
        return shrunk + self.epsilon * torch.eye(
            self.dimension,
            device=shrunk.device,
            dtype=shrunk.dtype,
        )

    def whiten(self, value: torch.Tensor) -> torch.Tensor:
        chol = torch.linalg.cholesky(self.matrix())
        value = value.to(chol.dtype)
        return torch.linalg.solve_triangular(
            chol, value.unsqueeze(-1), upper=False
        ).squeeze(-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        difference = left - right
        whitened = self.whiten(difference)
        distance = whitened.square().sum(-1)
        return distance / self.dimension if self.normalized else distance


class SquaredEuclideanMetric(nn.Module):
    """Squared Euclidean discrepancy with optional coordinate averaging."""

    def __init__(self, dimension: int, normalized: bool = True):
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.normalized = bool(normalized)
        self.register_buffer("identity", torch.eye(self.dimension))
        # Non-persistent so existing checkpoints keep loading with strict=True.
        self.register_buffer(
            "mean", torch.zeros(self.dimension), persistent=False
        )

    def update(
        self, states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        del states, mask

    def matrix(self) -> torch.Tensor:
        scale = 1.0 / self.dimension if self.normalized else 1.0
        return self.identity * scale

    def whiten(self, value: torch.Tensor) -> torch.Tensor:
        """Map into the metric's isotropic coordinates.

        Mirrors ``MahalanobisMetric.whiten`` so geometry exports can whiten
        under whichever metric a run used. Euclidean coordinates are already
        isotropic, so this is the identity: it preserves every pairwise
        distance ratio, which is all the t-SNE/UMAP export needs.
        """

        return value

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        distance = (left - right).square().sum(-1)
        return distance / self.dimension if self.normalized else distance


def terminal_set_discrepancy(
    state: torch.Tensor,
    goals: torch.Tensor,
    metric: EMAShrunkMahalanobis,
    temperature: float,
    goal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft minimum distance to one or more complete verified solution states."""
    if temperature <= 0:
        raise ValueError("terminal-set temperature must be positive")
    distance = metric(state.unsqueeze(-2), goals)
    if goal_mask is not None:
        distance = distance.masked_fill(~goal_mask, torch.inf)
        if bool((~goal_mask.any(-1)).any()):
            raise ValueError("every terminal set must contain a valid goal")
        count = goal_mask.sum(-1)
    else:
        count = torch.full(
            distance.shape[:-1], goals.shape[-2],
            device=distance.device, dtype=torch.long,
        )
    return -temperature * (
        torch.logsumexp(-distance / temperature, -1)
        - count.to(distance.dtype).log()
    )


def vicreg_floor_and_covariance(
    states: torch.Tensor,
    mask: torch.Tensor,
    variance_floor: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = states.reshape(-1, states.shape[-1])[mask.reshape(-1)]
    if len(flat) < 2:
        zero = states.sum() * 0
        return zero, zero
    centered = flat - flat.mean(0)
    covariance = centered.T @ centered / (len(flat) - 1)
    std = (covariance.diag() + epsilon).sqrt()
    variance = F.relu(variance_floor - std).square().mean()
    off_diagonal = covariance - covariance.diag().diag()
    decorrelation = off_diagonal.square().sum() / states.shape[-1]
    return variance, decorrelation


class SketchedIsotropicGaussianRegularizer(nn.Module):
    """SIGReg using the official sliced Epps--Pulley construction.

    The statistic follows LeJEPA's reference implementation: unit Gaussian
    directions, 17 trapezoidal characteristic-function points on [0, 3], and
    a deterministic new sketch at each call. This project retains its EMA and
    stop-gradient target path, so this is a regularizer ablation rather than a
    claim to reproduce the complete LeJEPA recipe.
    """

    def __init__(
        self,
        dimension: int,
        num_slices: int = 256,
        t_max: float = 3.0,
        num_points: int = 17,
    ):
        super().__init__()
        if dimension < 1 or num_slices < 1:
            raise ValueError("SIGReg dimensions and slices must be positive")
        if num_points < 3 or num_points % 2 != 1 or t_max <= 0:
            raise ValueError("invalid Epps--Pulley integration grid")
        self.dimension = int(dimension)
        self.num_slices = int(num_slices)
        t = torch.linspace(0, t_max, num_points, dtype=torch.float32)
        delta = t_max / (num_points - 1)
        weights = torch.full((num_points,), 2 * delta, dtype=torch.float32)
        weights[[0, -1]] = delta
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        samples = states.reshape(-1, states.shape[-1])[mask.reshape(-1)]
        if len(samples) < 2:
            return states.sum() * 0
        generator = torch.Generator(device=samples.device)
        generator.manual_seed(int(self.global_step))
        directions = torch.randn(
            self.dimension,
            self.num_slices,
            device=samples.device,
            dtype=samples.dtype,
            generator=generator,
        )
        directions = directions / directions.norm(dim=0).clamp_min(1e-12)
        self.global_step.add_(1)
        projected = (samples @ directions).float()
        phase = projected.unsqueeze(-1) * self.t
        cosine = phase.cos().mean(0)
        sine = phase.sin().mean(0)
        error = (cosine - self.phi).square() + sine.square()
        return ((error @ self.weights) * len(samples)).mean()


def dynamics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metric: EMAShrunkMahalanobis,
) -> torch.Tensor:
    return masked_mean(metric(prediction, target.detach()), mask)


def commutation_loss(
    token_endpoint: torch.Tensor,
    sentence_endpoint: torch.Tensor,
    mask: torch.Tensor,
    sentence_encoder: nn.Module,
    metric: EMAShrunkMahalanobis,
) -> torch.Tensor:
    """Compare nested token rollout and one-step sentence predictions."""
    return masked_mean(
        metric(sentence_encoder(token_endpoint), sentence_endpoint), mask
    )


def diagonal_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    variance_ratio = (posterior_logvar - prior_logvar).exp()
    squared = (
        (posterior_mean - prior_mean).square() * (-prior_logvar).exp()
    )
    per_dimension = 0.5 * (
        prior_logvar - posterior_logvar + variance_ratio + squared - 1
    )
    if free_bits:
        per_dimension = per_dimension.clamp_min(free_bits)
    return per_dimension.sum(-1)


def supported_step_cost(
    log_probability: torch.Tensor,
    step_cost: float,
    prior_weight: float,
) -> torch.Tensor:
    """``lambda_step - beta * log p(u | xi, q)``."""
    if step_cost < 0 or prior_weight < 0:
        raise ValueError("cost weights must be nonnegative")
    return step_cost - prior_weight * log_probability


def value_teacher_softmin(
    continuation_costs: torch.Tensor,
    continuation_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Soft-min over continuation samples and offline depths.

    The last dimensions may represent any rectangular sample/depth layout.
    They are teacher-computation axes, not value inputs.
    """
    if temperature <= 0:
        raise ValueError("teacher temperature must be positive")
    if continuation_costs.shape != continuation_mask.shape:
        raise ValueError("cost and continuation mask shapes differ")
    flat_cost = continuation_costs.flatten(2)
    flat_mask = continuation_mask.flatten(2)
    if bool((~flat_mask.any(-1)).any()):
        raise ValueError("every first action needs a continuation")
    scaled = (-flat_cost / temperature).masked_fill(~flat_mask, -torch.inf)
    return -temperature * torch.logsumexp(scaled, -1)


def listwise_value_loss(
    teacher_cost: torch.Tensor,
    predicted_cost: torch.Tensor,
    action_mask: torch.Tensor,
    teacher_temperature: float,
    value_temperature: float,
) -> torch.Tensor:
    """Within-root KL ranking loss used by the unbudgeted top-level value."""
    if teacher_cost.shape != predicted_cost.shape or action_mask.shape != teacher_cost.shape:
        raise ValueError("value-ranking tensors must have identical shapes")
    if teacher_temperature <= 0 or value_temperature <= 0:
        raise ValueError("ranking temperatures must be positive")
    if bool((~action_mask.any(-1)).any()):
        raise ValueError("every root must contain at least one valid action")
    teacher_logits = (-teacher_cost / teacher_temperature).masked_fill(
        ~action_mask, -torch.inf
    )
    predicted_logits = (-predicted_cost / value_temperature).masked_fill(
        ~action_mask, -torch.inf
    )
    teacher = torch.softmax(teacher_logits, -1).detach()
    return F.kl_div(
        torch.log_softmax(predicted_logits, -1), teacher,
        reduction="batchmean",
    )


def recursive_rollout_loss(
    predictor: nn.Module,
    start: torch.Tensor,
    actions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    metric: EMAShrunkMahalanobis,
    *,
    truncate_bptt: int | None = None,
) -> torch.Tensor:
    """Sparse predicted-input loss for N=2/4/8 rollout batches."""
    if actions.shape[:2] != targets.shape[:2] or mask.shape != actions.shape[:2]:
        raise ValueError("rollout tensors do not align")
    if truncate_bptt is None:
        prediction, _ = predictor.rollout(start, actions)
        return masked_mean(metric(prediction, targets.detach()), mask)
    if truncate_bptt < 1:
        raise ValueError("truncate_bptt must be positive")
    # Preserve the root cache within each truncated segment. Detach the state
    # history only at explicit BPTT boundaries.
    state_history = start[:, None]
    action_history = actions[:, :0]
    predictions = []
    for step in range(actions.shape[1]):
        segment_action = actions[:, step:step + 1]
        rolled, _ = predictor.rollout(
            state_history[:, -1], segment_action,
            state_history=state_history,
            action_history=action_history,
        )
        current = rolled[:, -1]
        predictions.append(current)
        state_history = torch.cat([state_history, current[:, None]], 1)
        action_history = torch.cat([action_history, segment_action], 1)
        if (step + 1) % truncate_bptt == 0:
            state_history = state_history.detach()
            action_history = action_history.detach()
    return masked_mean(
        metric(torch.stack(predictions, 1), targets.detach()), mask
    )
