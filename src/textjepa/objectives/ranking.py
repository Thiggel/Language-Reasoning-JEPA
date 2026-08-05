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


class CostRanking(Objective):
    """Depth-calibrated ranking: order full MPC costs (depth + V) across
    search depths, not just 1-step values. Compares the executed 2-step
    continuation's cost (2 + V(F(F(s,a_t),a_{t+1}))) against 1-step
    alternatives' costs (1 + V(F(s,alt))) with symbolic cost targets.
    Fixes the look-2 anomaly: plain ranking perfects 1-step order while
    distorting the absolute scale that multi-step cost sums rely on."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "exec2_value" not in out.extras:
            return out.step_states.sum() * 0.0
        c2 = 2.0 + out.extras["exec2_value"]  # [B, T-1]
        t2 = 2.0 + batch["remaining"][:, 1:].float()
        c1 = 1.0 + out.extras["alt_value"][:, :-1]  # [B, T-1, K]
        t1 = 1.0 + batch["alt_remaining"][:, :-1].float()
        valid = (batch["alt_remaining"][:, :-1] >= 0) & out.step_mask[
            :, 1:
        ].unsqueeze(-1)
        two_better = (t2.unsqueeze(-1) < t1) & valid
        one_better = (t2.unsqueeze(-1) > t1) & valid
        diff = c2.unsqueeze(-1) - c1
        loss = two_better.float() * F.relu(self.margin + diff) + (
            one_better.float() * F.relu(self.margin - diff)
        )
        n = (two_better | one_better).float().sum().clamp(min=1.0)
        return loss.sum() / n


class GeoAdvantageRank(Objective):
    """Annotation-free counterfactual ranking: order V(F(s,a_i)) by the
    GEOMETRIC quality of each action's true next state (LN-L1 distance of
    the EMA-encoded outcome text to the EMA terminal goal). Environment
    interaction only — no symbolic labels."""

    def __init__(self, margin: float = 0.5, label_gap: float = 0.02):
        super().__init__()
        self.margin = margin
        self.label_gap = label_gap

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "ga_energy" not in out.extras:
            return out.step_states.sum() * 0.0
        e = out.extras["ga_energy"]  # [B, 1+K]
        d = out.extras["ga_label"]
        v = out.extras["ga_valid"]
        di = d.unsqueeze(2) - d.unsqueeze(1)  # label diffs
        ei = e.unsqueeze(2) - e.unsqueeze(1)
        pair_v = v.unsqueeze(2) & v.unsqueeze(1)
        better = (di < -self.label_gap) & pair_v  # i closer to goal than j
        loss = better.float() * F.relu(self.margin + ei)
        n = better.float().sum().clamp(min=1.0)
        return loss.sum() / n


class GeoAdvantageRegression(Objective):
    """Calibrate geometric action advantages without an action-only head.

    ``ga_energy[i]`` is always computed as ``V(F(s, a_i), g)``.  For each
    same-state candidate pair, the predicted advantage of action ``i`` over
    ``j`` is therefore ``energy[j] - energy[i]``.  Its target is the matching
    difference between true EMA-geometry distances.  Pair differences remove
    arbitrary state-specific offsets while retaining the magnitude that a
    ranking loss discards.
    """

    def __init__(self, target_scale: float = 1.0):
        super().__init__()
        if target_scale <= 0:
            raise ValueError("target_scale must be positive")
        self.target_scale = float(target_scale)

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "ga_energy" not in out.extras:
            return out.step_states.sum() * 0.0
        energy = out.extras["ga_energy"]
        valid = out.extras["ga_valid"]
        # Padded or failed rollout candidates deliberately carry +inf
        # distances.  Zero them *before* constructing pair differences;
        # multiplying an inf/nan error by a false mask afterwards is not
        # numerically safe.
        distance = out.extras["ga_label"].detach().masked_fill(~valid, 0.0)
        count = energy.shape[1]
        upper = torch.triu(
            torch.ones(count, count, dtype=torch.bool, device=energy.device),
            diagonal=1,
        )
        pair_valid = (
            valid.unsqueeze(2) & valid.unsqueeze(1) & upper.unsqueeze(0)
        )
        predicted_advantage = energy.unsqueeze(1) - energy.unsqueeze(2)
        target_advantage = self.target_scale * (
            distance.unsqueeze(1) - distance.unsqueeze(2)
        )
        squared_error = (predicted_advantage - target_advantage).square()
        return squared_error[pair_valid].sum() / pair_valid.sum().clamp(min=1)


class GeoEnergyRegression(Objective):
    """Calibrate each lower-is-better Energy to its absolute teacher target."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "ga_energy_target" not in out.extras:
            return out.step_states.sum() * 0.0
        energy = out.extras["ga_energy"]
        target = out.extras["ga_energy_target"].detach()
        valid = out.extras["ga_valid"] & torch.isfinite(target)
        safe_target = target.masked_fill(~valid, 0.0)
        error = (energy - safe_target).square()
        return error[valid].sum() / valid.sum().clamp(min=1)


class GeoRolloutEnergyRegression(Objective):
    """Calibrate state Energy on recursively imagined factual rollouts."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        energies = out.extras.get("ga_rollout_state_energy", ())
        if not energies:
            return out.step_states.sum() * 0.0
        targets = out.extras["ga_rollout_state_energy_target"]
        masks = out.extras["ga_rollout_state_energy_mask"]
        losses = []
        for energy, target, valid in zip(energies, targets, masks):
            target = target.detach()
            valid = valid & torch.isfinite(target)
            safe_target = target.masked_fill(~valid, 0.0)
            squared_error = (energy - safe_target).square()
            losses.append(
                squared_error[valid].sum() / valid.sum().clamp(min=1)
            )
        return torch.stack(losses).mean()


class GeoRolloutAdvantageRank(Objective):
    """Apply the ordinary GAR ordering to candidates from drifted anchors."""

    def __init__(self, margin: float = 0.5, label_gap: float = 0.02):
        super().__init__()
        self.margin = margin
        self.label_gap = label_gap

    def forward(self, out, batch: dict) -> torch.Tensor:
        energies = out.extras.get("ga_rank_rollout_energy", ())
        if not energies:
            return out.step_states.sum() * 0.0
        losses = []
        for energy, label, valid in zip(
            energies,
            out.extras["ga_rank_rollout_label"],
            out.extras["ga_rank_rollout_valid"],
        ):
            label_diff = label.unsqueeze(2) - label.unsqueeze(1)
            energy_diff = energy.unsqueeze(2) - energy.unsqueeze(1)
            pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
            better = (label_diff < -self.label_gap) & pair_valid
            loss = better.float() * F.relu(self.margin + energy_diff)
            losses.append(loss.sum() / better.sum().clamp(min=1))
        return torch.stack(losses).mean()


class GeoRolloutAdvantageRegression(Objective):
    """Calibrate pairwise Energy differences at drifted GAR anchors."""

    def __init__(self, target_scale: float = 1.0):
        super().__init__()
        if target_scale <= 0:
            raise ValueError("target_scale must be positive")
        self.target_scale = float(target_scale)

    def forward(self, out, batch: dict) -> torch.Tensor:
        energies = out.extras.get("ga_rank_rollout_energy", ())
        if not energies:
            return out.step_states.sum() * 0.0
        losses = []
        for energy, label, valid in zip(
            energies,
            out.extras["ga_rank_rollout_label"],
            out.extras["ga_rank_rollout_valid"],
        ):
            # Failed or padded candidates carry +inf teacher distances.  Make
            # them finite before constructing pair differences: indexing the
            # valid pairs only after ``inf - inf`` has already produced NaNs
            # can still poison autograd.
            label = label.detach().masked_fill(~valid, 0.0)
            count = energy.shape[1]
            upper = torch.triu(
                torch.ones(
                    count, count, dtype=torch.bool, device=energy.device
                ),
                diagonal=1,
            )
            pair_valid = (
                valid.unsqueeze(2) & valid.unsqueeze(1) & upper.unsqueeze(0)
            )
            predicted = energy.unsqueeze(1) - energy.unsqueeze(2)
            target = self.target_scale * (
                label.unsqueeze(1) - label.unsqueeze(2)
            )
            error = (predicted - target).square()
            losses.append(
                error[pair_valid].sum() / pair_valid.sum().clamp(min=1)
            )
        return torch.stack(losses).mean()


class GeoRolloutCandidateEnergyRegression(Objective):
    """Regress absolute distance or transition change at drifted anchors."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        energies = out.extras.get("ga_rank_rollout_energy", ())
        if not energies:
            return out.step_states.sum() * 0.0
        losses = []
        for energy, target, valid in zip(
            energies,
            out.extras["ga_rank_rollout_target"],
            out.extras["ga_rank_rollout_valid"],
        ):
            target = target.detach()
            valid = valid & torch.isfinite(target)
            safe_target = target.masked_fill(~valid, 0.0)
            error = (energy - safe_target).square()
            losses.append(error[valid].sum() / valid.sum().clamp(min=1))
        return torch.stack(losses).mean()
