"""Trajectory-geometry regularizers.

TemporalStraightening (arXiv:2603.12231): maximize cosine similarity of
consecutive latent velocities v_t = s_{t+1} - s_t so that Euclidean latent
distance approximates geodesic (minimum-step) distance — the property that
makes raw-geometry goal-distance planning work. Label-free, applied to
online states.

GoalMonotonicity: hinge on the LN-L1 distance to the trace-terminal EMA
state — necessary steps must strictly decrease it, distractor steps must
not decrease it. Supervised by the same symbolic labels as the value head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, masked_mean


def _all_states(out) -> torch.Tensor:
    return torch.cat([out.s0.unsqueeze(1), out.step_states], dim=1)


def velocity_cosines(out) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine of consecutive velocities [B, T-1] and its validity mask."""
    v = torch.diff(_all_states(out), dim=1)  # [B, T, D]
    cos = F.cosine_similarity(v[:, :-1], v[:, 1:], dim=-1)
    return cos, out.step_mask[:, 1:].float()


def goal_distances(out) -> torch.Tensor:
    """LN-L1 distance of every state (incl. s0) to its trace-terminal EMA
    target state; [B, T+1]."""
    B = out.s0.shape[0]
    last = out.step_mask.sum(dim=1).clamp(min=1) - 1
    goal = out.step_states_tgt[torch.arange(B, device=out.s0.device), last]
    ln = lambda x: F.layer_norm(x, x.shape[-1:])
    return (ln(_all_states(out)) - ln(goal).unsqueeze(1)).abs().mean(-1)


class TemporalStraightening(Objective):
    def forward(self, out, batch: dict) -> torch.Tensor:
        cos, mask = velocity_cosines(out)
        return masked_mean(1.0 - cos, mask)


class GoalMonotonicity(Objective):
    def __init__(self, margin: float = 0.02, distractor_weight: float = 0.5):
        super().__init__()
        self.margin = margin
        self.distractor_weight = distractor_weight

    def forward(self, out, batch: dict) -> torch.Tensor:
        d = goal_distances(out)  # [B, T+1]
        delta = d[:, 1:] - d[:, :-1]  # <0 means the step moved toward goal
        nec = batch["necessary"].float()
        loss = nec * F.relu(delta + self.margin) + (
            self.distractor_weight * (1 - nec) * F.relu(-delta)
        )
        return masked_mean(loss, out.step_mask.float())
