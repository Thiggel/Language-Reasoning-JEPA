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
