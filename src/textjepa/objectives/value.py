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
