"""Goal-energy supervision: predict remaining necessary steps per state."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, masked_mean


class ValueRegression(Objective):
    def forward(self, out, batch: dict) -> torch.Tensor:
        remaining = torch.cat(
            [batch["n_necessary"].unsqueeze(1), batch["remaining"]], dim=1
        ).float()
        mask = torch.cat(
            [torch.ones_like(out.step_mask[:, :1]), out.step_mask], dim=1
        ).float()
        err = F.smooth_l1_loss(out.value_pred, remaining, reduction="none")
        return masked_mean(err, mask)


class ValueDistill(Objective):
    """Self-distill the latent goal distance into the value head: V(s_t, s_0)
    regresses onto d_goal(s_t) to the trace-terminal EMA state. Label-free —
    no symbolic remaining-steps supervision; uses the geometry projection
    when the model has one."""

    def __init__(self, scale: float = 5.0):
        super().__init__()
        # d_goal lives in LN-L1 units (~0.1-1.5); scale to remaining-steps
        # magnitude so the planner's cost mixing (steps + V) stays sane
        self.scale = scale

    def forward(self, out, batch: dict) -> torch.Tensor:
        from textjepa.objectives.geometry import goal_distances

        target = (goal_distances(out) * self.scale).detach()
        mask = torch.cat(
            [torch.ones_like(out.step_mask[:, :1]), out.step_mask], dim=1
        ).float()
        err = F.smooth_l1_loss(out.value_pred, target, reduction="none")
        return masked_mean(err, mask)


class GoalAdvantageDistill(Objective):
    """Distill privileged terminal-distance improvement into V(state, action).

    A positive target means that executing the action and then following the
    observed continuation for the configured horizon moved closer to the EMA
    terminal representation.  The goal is never an input to the learned head.
    """

    def __init__(
        self,
        regression_weight: float = 1.0,
        pairwise_weight: float = 0.0,
        margin: float = 0.5,
        label_gap: float = 0.001,
    ):
        super().__init__()
        self.regression_weight = float(regression_weight)
        self.pairwise_weight = float(pairwise_weight)
        self.margin = float(margin)
        self.label_gap = float(label_gap)

    def forward(self, out, batch: dict) -> torch.Tensor:
        prediction = out.extras.get("gar_action_value")
        if prediction is None:
            return out.preds.sum() * 0.0
        target = out.extras["gar_action_target"]
        error = F.smooth_l1_loss(prediction, target, reduction="none")
        expert = masked_mean(error, out.step_mask.float())
        alt_prediction = out.extras.get("gar_alt_action_value")
        if alt_prediction is None:
            return self.regression_weight * expert
        alt_error = F.smooth_l1_loss(
            alt_prediction, out.extras["gar_alt_action_target"], reduction="none"
        )
        alternatives = masked_mean(
            alt_error, out.extras["gar_alt_action_valid"].float()
        )
        # Candidate breadth must not silently increase GAR's regression scale.
        regression = 0.5 * (expert + alternatives)
        if self.pairwise_weight == 0.0:
            return self.regression_weight * regression

        predictions = torch.cat(
            [prediction.unsqueeze(-1), alt_prediction], dim=-1
        )
        targets = torch.cat([
            target.unsqueeze(-1), out.extras["gar_alt_action_target"]
        ], dim=-1)
        valid = torch.cat([
            out.step_mask.unsqueeze(-1),
            out.extras["gar_alt_action_valid"],
        ], dim=-1)
        target_delta = targets.unsqueeze(-1) - targets.unsqueeze(-2)
        prediction_delta = (
            predictions.unsqueeze(-1) - predictions.unsqueeze(-2)
        )
        # Positive target_delta means the row action makes more progress than
        # the column action, so its learned value must be larger by margin.
        better = (
            valid.unsqueeze(-1) & valid.unsqueeze(-2)
            & target_delta.gt(self.label_gap)
        )
        pairwise = masked_mean(
            F.relu(self.margin - prediction_delta), better.float()
        )
        return (
            self.regression_weight * regression
            + self.pairwise_weight * pairwise
        )


class ActionKL(Objective):
    """KL(q(a|s,s') || p(a|s)) for variational unobserved actions.
    ``free_nats``: KL below this threshold is not penalized (free bits) —
    prevents posterior collapse."""

    def __init__(self, free_nats: float = 0.0):
        super().__init__()
        self.free_nats = free_nats

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "action_kl" not in out.extras:
            return out.step_states.sum() * 0.0
        kl = out.extras["action_kl"]
        if self.free_nats:
            kl = torch.clamp(kl - self.free_nats, min=0.0)
        return masked_mean(kl, out.step_mask.float())


class ActionDecode(Objective):
    """Detached readout: latent action code -> frozen intent-anchor
    embedding (interpretability + plan-time action matching; gradients do
    not reach the code)."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "act_decode" not in out.extras:
            return out.step_states.sum() * 0.0
        from textjepa.objectives.base import latent_distance

        d = latent_distance(
            out.extras["act_decode"], out.extras["act_decode_tgt"],
            "smooth_l1", True,
        )
        return masked_mean(d, out.step_mask.float())
