"""VICReg, SIGReg, and VISReg stabilization on online latents."""

from __future__ import annotations

import math

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
        if "sigreg_states" in out.extras:
            source = out.extras["sigreg_states"]
            source_mask = out.extras["sigreg_state_mask"]
            states = source.reshape(-1, source.shape[-1])[
                source_mask.reshape(-1)
            ]
        else:
            mask = out.step_mask.reshape(-1)
            states = torch.cat([
                out.s0,
                out.step_states.reshape(-1, out.step_states.shape[-1])[mask],
            ])
        var_s, cov_s = variance_covariance(states, self.std_target)
        loss = var_s + self.cov_weight * cov_s
        if self.action_weight > 0:
            acts = out.actions.reshape(-1, out.actions.shape[-1])[mask]
            var_a, _ = variance_covariance(acts, self.std_target)
            loss = loss + self.action_weight * var_a
        return loss


class MultiscaleVICReg(Objective):
    """Separate variance/covariance gates for active token and sentence spaces."""

    def __init__(self, std_target: float = 1.0, cov_weight: float = 0.04,
                 action_weight: float = 0.1):
        super().__init__()
        self.std_target = float(std_target)
        self.cov_weight = float(cov_weight)
        self.action_weight = float(action_weight)

    def _space_loss(self, value, mask):
        flat = value.reshape(-1, value.shape[-1])[mask.reshape(-1)]
        var, cov = variance_covariance(flat, self.std_target)
        return var + self.cov_weight * cov

    def forward(self, out, batch: dict) -> torch.Tensor:
        losses = []
        if out.extras.get("token_predictions") is not None:
            losses.append(self._space_loss(
                out.extras["token_states"], out.extras["token_state_mask"]
            ))
        if out.extras.get("sentence_predictions") is not None:
            sentence_mask = out.extras["sentence_states"].abs().sum(-1).gt(0)
            losses.append(self._space_loss(
                out.extras["sentence_states"], sentence_mask
            ))
        if not losses:
            return out.preds.sum() * 0.0
        loss = torch.stack(losses).mean()
        if self.action_weight:
            actions = out.actions.reshape(-1, out.actions.shape[-1])[
                out.step_mask.reshape(-1)
            ]
            var, _ = variance_covariance(actions, self.std_target)
            loss = loss + self.action_weight * var
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
        if "sigreg_states" in out.extras:
            source = out.extras["sigreg_states"]
            source_mask = out.extras["sigreg_state_mask"]
            x = source.reshape(-1, source.shape[-1])[
                source_mask.reshape(-1)
            ].float()
        else:
            mask = out.step_mask.reshape(-1)
            x = torch.cat([
                out.s0,
                out.step_states.reshape(-1, out.step_states.shape[-1])[mask],
            ], dim=0).float()
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


class VISReg(Objective):
    """Faithful variance-invariance-sketching regularizer.

    This follows the authors' official implementation for arXiv:2606.02572:
    centering, squared unit-scale matching, and sliced-Wasserstein matching of
    scale-normalized random projections to standard-normal quantiles.  The
    standard deviation is detached only in the shape term so scale and shape
    gradients remain decoupled; target encodings themselves are not detached.
    """

    def __init__(self, num_projections: int = 256):
        super().__init__()
        self.num_projections = int(num_projections)
        self._cached_batch = -1
        self._cached_target: torch.Tensor | None = None

    def _target(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self._cached_batch != batch_size or self._cached_target is None:
            quantiles = torch.linspace(
                1, batch_size, batch_size, device=device, dtype=torch.float32
            ) / (batch_size + 1)
            self._cached_target = torch.erfinv(2 * quantiles - 1).mul_(
                math.sqrt(2)
            )
            self._cached_batch = batch_size
        return self._cached_target.to(device=device, dtype=dtype)

    def forward(self, out, batch: dict) -> torch.Tensor:
        del batch
        if "sigreg_states" in out.extras:
            source = out.extras["sigreg_states"]
            source_mask = out.extras["sigreg_state_mask"]
            z = source.reshape(-1, source.shape[-1])[
                source_mask.reshape(-1)
            ].float()
        else:
            mask = out.step_mask.reshape(-1)
            z = torch.cat([
                out.s0,
                out.step_states.reshape(-1, out.step_states.shape[-1])[mask],
            ], dim=0).float()
        # Official code accepts [views, batch, dimensions]. Our independent
        # transition batch is one view of each current state.
        z = z.unsqueeze(0)
        _, batch_size, dimensions = z.shape
        mean = z.mean(dim=1, keepdim=True)
        center_loss = mean.square().mean()
        centered = z - mean
        std = centered.norm(dim=1).div(math.sqrt(batch_size)) + 1e-6
        scale_loss = (std - 1.0).square().mean()
        normalized = centered / std.detach().unsqueeze(1)
        directions = torch.randn(
            dimensions, self.num_projections,
            device=z.device, dtype=z.dtype,
        )
        directions = directions / directions.norm(
            dim=0, keepdim=True
        ).clamp_min(1e-12)
        projected = (normalized @ directions).sort(dim=1).values
        target = self._target(
            batch_size, z.device, z.dtype
        ).view(1, batch_size, 1)
        shape_loss = (projected - target).square().mean()
        return scale_loss + shape_loss + center_loss
