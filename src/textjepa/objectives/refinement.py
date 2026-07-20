"""Behavioral action prior for replacement-only iterative refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective


class RefinementActionPrior(Objective):
    """Supervise pointer and replacement-token distributions separately."""

    def __init__(self, position_weight: float = 1.0,
                 content_weight: float = 1.0):
        super().__init__()
        self.position_weight = float(position_weight)
        self.content_weight = float(content_weight)

    def forward(self, out, batch: dict) -> torch.Tensor:
        position_logits = out.extras["refinement_position_logits"]
        content_logits = out.extras["refinement_content_logits"]
        steps = position_logits.shape[1]
        valid = out.step_mask[:, :steps] & batch["op"][:, :steps].eq(2)
        if not bool(valid.any()):
            return position_logits.sum() * 0.0
        position = F.cross_entropy(
            position_logits[valid], batch["edit_position"][:, :steps][valid]
        )
        content = F.cross_entropy(
            content_logits[valid], batch["edit_content_token"][:, :steps][valid]
        )
        return self.position_weight * position + self.content_weight * content
