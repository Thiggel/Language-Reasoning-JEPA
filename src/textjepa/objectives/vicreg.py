"""VICReg-style variance/covariance stabilization on online latents."""

from __future__ import annotations

import torch

from textjepa.objectives.base import Objective


def variance_covariance(x: torch.Tensor, std_target: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [N, D] -> (variance hinge, off-diagonal covariance penalty)."""
    x = x - x.mean(dim=0)
    std = torch.sqrt(x.var(dim=0) + 1e-4)
    var_loss = torch.relu(std_target - std).mean()
    n = max(x.shape[0] - 1, 1)
    cov = (x.T @ x) / n
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / x.shape[1]
    return var_loss, cov_loss


class VICReg(Objective):
    """Applies variance/covariance terms to states (and optionally actions)."""

    def __init__(
        self,
        std_target: float = 1.0,
        cov_weight: float = 0.04,
        action_weight: float = 0.1,
    ):
        super().__init__()
        self.std_target = std_target
        self.cov_weight = cov_weight
        self.action_weight = action_weight

    def forward(self, out, batch: dict) -> torch.Tensor:
        mask = out.step_mask.reshape(-1)
        states = torch.cat(
            [out.s0, out.step_states.reshape(-1, out.step_states.shape[-1])[mask]]
        )
        var_s, cov_s = variance_covariance(states, self.std_target)
        loss = var_s + self.cov_weight * cov_s
        if self.action_weight > 0:
            acts = out.actions.reshape(-1, out.actions.shape[-1])[mask]
            var_a, _ = variance_covariance(acts, self.std_target)
            loss = loss + self.action_weight * var_a
        return loss


class SIGReg(Objective):
    """Sketched Epps--Pulley test against an isotropic Gaussian.

    This follows LeJEPA Algorithm 1: random unit projections, empirical
    characteristic functions at 17 integration points, a Gaussian window,
    and trapezoidal quadrature.  Directions are resampled on every call.
    """

    def __init__(self, num_slices: int = 256, n_points: int = 17,
                 integration_limit: float = 5.0):
        super().__init__()
        self.num_slices = num_slices
        self.n_points = n_points
        self.integration_limit = integration_limit

    def forward(self, out, batch: dict) -> torch.Tensor:
        mask = out.step_mask.reshape(-1)
        x = torch.cat([
            out.s0,
            out.step_states.reshape(-1, out.step_states.shape[-1])[mask],
        ], dim=0)
        directions = torch.randn(
            x.shape[-1], self.num_slices, device=x.device, dtype=x.dtype
        )
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
        t = torch.linspace(
            -self.integration_limit, self.integration_limit, self.n_points,
            device=x.device, dtype=x.dtype,
        )
        projected = (x @ directions).unsqueeze(-1) * t
        ecf = torch.complex(projected.cos(), projected.sin()).mean(0)
        normal_cf = torch.exp(-0.5 * t.square())
        err = (ecf - normal_cf).abs().square() * normal_cf
        statistic = torch.trapz(err, t, dim=-1) * x.shape[0]
        return statistic.mean()
