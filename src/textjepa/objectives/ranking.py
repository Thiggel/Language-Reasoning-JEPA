"""Counterfactual action ranking.

For each visited state the batch carries K alternative feasible actions
with their ground-truth outcome quality (remaining necessary steps /
defects after the action). The core predicts and value-scores executed
and alternative actions; this loss enforces a margin between every pair
whose outcomes differ — the executed action must beat worse alternatives
and lose to strictly better ones. Directly targets energy ties, which
regression losses tolerate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective


class ActionRanking(Objective):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "alt_value" not in out.extras:
            return out.step_states.sum() * 0.0
        e_exec = out.extras["exec_value"].unsqueeze(-1)  # [B, T, 1]
        e_alt = out.extras["alt_value"]  # [B, T, K]
        r_exec = batch["remaining"].float().unsqueeze(-1)
        r_alt = batch["alt_remaining"].float()
        valid = (r_alt >= 0) & out.step_mask.unsqueeze(-1)
        exec_better = (r_exec < r_alt) & valid
        alt_better = (r_exec > r_alt) & valid
        diff = e_exec - e_alt  # want negative when exec is better
        loss = exec_better.float() * F.relu(self.margin + diff) + (
            alt_better.float() * F.relu(self.margin - diff)
        )
        n = (exec_better | alt_better).float().sum().clamp(min=1.0)
        return loss.sum() / n
