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
