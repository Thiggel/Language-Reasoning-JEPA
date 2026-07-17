"""Latent prediction losses: teacher-forced, open-loop rollout, hierarchy."""

from __future__ import annotations

import torch

from textjepa.objectives.base import Objective, latent_distance, masked_mean


class LatentPrediction(Objective):
    """||F(s_t, a_t) - sg(s̄_{t+1})|| over valid steps (teacher forcing)."""

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind, self.norm_targets = kind, norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        d = latent_distance(out.preds, out.step_states_tgt, self.kind, self.norm_targets)
        return masked_mean(d, out.step_mask.float())


class TokenAlignedPrediction(Objective):
    """Next-token-latent JEPA loss on the structured edit state."""

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind, self.norm_targets = kind, norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        pred = out.extras.get("token_predictions")
        if pred is None:
            return out.preds.sum() * 0.0
        target = out.extras["token_targets"]
        mask = (
            out.extras["token_prediction_mask"]
            & out.extras["token_target_mask"]
            & out.step_mask.unsqueeze(-1)
        )
        distance = latent_distance(
            pred, target, self.kind, self.norm_targets
        )
        return masked_mean(distance, mask.float())


class TokenAlignedRolloutPrediction(Objective):
    """Deep supervision of recursive token-state rollouts from the first state."""

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True,
                 max_depth: int = 0):
        super().__init__()
        self.kind, self.norm_targets = kind, norm_targets
        self.max_depth = int(max_depth)

    def forward(self, out, batch: dict) -> torch.Tensor:
        pred = out.extras.get("token_rollout_predictions")
        if pred is None:
            return out.preds.sum() * 0.0
        target = out.extras["token_targets"]
        mask = (
            out.extras["token_rollout_mask"]
            & out.extras["token_target_mask"]
            & out.step_mask.unsqueeze(-1)
        )
        if self.max_depth:
            depth = torch.arange(mask.shape[1], device=mask.device)
            mask = mask & (depth < self.max_depth).view(1, -1, 1)
        distance = latent_distance(
            pred, target, self.kind, self.norm_targets
        )
        return masked_mean(distance, mask.float())


class RolloutPrediction(Objective):
    """Open-loop rollout from s0 through teacher actions vs EMA targets.

    ``max_depth``: supervise only the first N rollout steps (0 = all) —
    the supervision-horizon ablation."""

    def __init__(
        self, kind: str = "smooth_l1", norm_targets: bool = True,
        max_depth: int = 0,
    ):
        super().__init__()
        self.kind, self.norm_targets = kind, norm_targets
        self.max_depth = max_depth

    def forward(self, out, batch: dict) -> torch.Tensor:
        d = latent_distance(out.rollout, out.step_states_tgt, self.kind, self.norm_targets)
        mask = out.step_mask.float()
        if self.max_depth:
            T = mask.shape[1]
            depth_ok = (
                torch.arange(T, device=mask.device) < self.max_depth
            ).float().unsqueeze(0)
            mask = mask * depth_ok
        return masked_mean(d, mask)


class DenseRolloutPrediction(Objective):
    """Open-loop loss from every valid origin at every horizon up to N."""

    def __init__(
        self,
        kind: str = "smooth_l1",
        norm_targets: bool = True,
        horizon_discount: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets
        self.horizon_discount = horizon_discount

    def forward(self, out, batch: dict) -> torch.Tensor:
        predictions = out.extras.get("dense_rollout_predictions")
        if not predictions:
            return out.preds.sum() * 0.0
        targets = out.extras["dense_rollout_targets"]
        masks = out.extras["dense_rollout_masks"]
        losses = []
        weights = []
        for horizon, (prediction, target, mask) in enumerate(
            zip(predictions, targets, masks)
        ):
            distance = latent_distance(
                prediction, target.detach(), self.kind, self.norm_targets
            )
            weight = self.horizon_discount ** horizon
            losses.append(weight * masked_mean(distance, mask.float()))
            weights.append(weight)
        return torch.stack(losses).sum() / sum(weights)


class HierarchyPrediction(Objective):
    """||F_hi(s_t, macro(a_{t:t+K})) - sg(s̄_{t+K})|| over valid windows."""

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind, self.norm_targets = kind, norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        if out.hi_preds is None:
            return torch.zeros((), device=out.preds.device)
        d = latent_distance(out.hi_preds, out.hi_targets, self.kind, self.norm_targets)
        return masked_mean(d, out.hi_mask.float())


class DenseHierarchyRolloutPrediction(Objective):
    """Planning-matched recursive macro rollout loss from every origin."""

    def __init__(
        self,
        kind: str = "smooth_l1",
        norm_targets: bool = True,
        horizon_discount: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets
        self.horizon_discount = horizon_discount

    def forward(self, out, batch: dict) -> torch.Tensor:
        predictions = out.extras.get("high_dense_rollout_predictions")
        if not predictions:
            return out.preds.sum() * 0.0
        targets = out.extras["high_dense_rollout_targets"]
        masks = out.extras["high_dense_rollout_masks"]
        losses = []
        weights = []
        for horizon, (prediction, target, mask) in enumerate(
            zip(predictions, targets, masks)
        ):
            distance = latent_distance(
                prediction, target.detach(), self.kind, self.norm_targets
            )
            weight = self.horizon_discount ** horizon
            losses.append(weight * masked_mean(distance, mask.float()))
            weights.append(weight)
        return torch.stack(losses).sum() / sum(weights)


class MacroPrior(Objective):
    """Fit the state-conditioned macro prior or variational q/p pair."""

    def __init__(self, free_nats: float = 0.0):
        super().__init__()
        self.free_nats = free_nats

    def forward(self, out, batch: dict) -> torch.Tensor:
        loss = out.extras.get("macro_prior_loss")
        if loss is None or out.hi_mask is None:
            return out.preds.sum() * 0.0
        if self.free_nats:
            loss = torch.clamp(loss - self.free_nats, min=0.0)
        return masked_mean(loss, out.hi_mask.float())


class HierarchyValueDistill(Objective):
    """Train the high-level value on geometric waypoint advantages."""

    def __init__(self, scale: float = 5.0):
        super().__init__()
        self.scale = scale

    def forward(self, out, batch: dict) -> torch.Tensor:
        pred = out.extras.get("hi_value_pred")
        target = out.extras.get("hi_value_target")
        if pred is None or target is None or out.hi_mask is None:
            return out.preds.sum() * 0.0
        loss = torch.nn.functional.smooth_l1_loss(
            pred, target.detach() * self.scale, reduction="none"
        )
        return masked_mean(loss, out.hi_mask.float())


class HierarchyValueRegression(Objective):
    """Exact remaining-step supervision on predicted macro states."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        pred = out.extras.get("hi_value_pred")
        target = out.extras.get("hi_remaining_target")
        if pred is None or target is None or out.hi_mask is None:
            return out.preds.sum() * 0.0
        loss = torch.nn.functional.smooth_l1_loss(
            pred, target.detach(), reduction="none"
        )
        return masked_mean(loss, out.hi_mask.float())


class DenseHierarchyValueRegression(Objective):
    """Value supervision on recursively predicted planning-time states."""

    def __init__(self, horizon_discount: float = 1.0):
        super().__init__()
        self.horizon_discount = horizon_discount

    def forward(self, out, batch: dict) -> torch.Tensor:
        predictions = out.extras.get("high_dense_value_predictions")
        if not predictions:
            return out.preds.sum() * 0.0
        targets = out.extras["high_dense_value_targets"]
        masks = out.extras["high_dense_rollout_masks"]
        losses = []
        weights = []
        for horizon, (prediction, target, mask) in enumerate(
            zip(predictions, targets, masks)
        ):
            loss = torch.nn.functional.smooth_l1_loss(
                prediction, target.detach(), reduction="none"
            )
            weight = self.horizon_discount ** horizon
            losses.append(weight * masked_mean(loss, mask.float()))
            weights.append(weight)
        return torch.stack(losses).sum() / sum(weights)
