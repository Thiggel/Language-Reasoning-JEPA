"""Energy-head counterfactual feasibility ranking (self-supervised)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, masked_mean


class EnergyCFFeasibilityRank(Objective):
    """Observed continuation beats every counterfactual under the planning
    Energy.

    For each ranking anchor (s_t, u_t, s_{t+1}) and its counterfactual
    actions u' (the batch's ranking alternatives, including the hard
    infeasible negatives from ``invalid_counterfactual_k``), the model emits
    E_obs = E(predictor(s_t, u_t)) and E_cf = E(predictor(s_t, u')) through
    the SAME Energy head planning scores with (``DiscourseJEPA.
    _energy_cf_feasibility``).  Loss = softplus(E_obs - E_cf) averaged over
    valid pairs: logistic pairwise, observed must have LOWER energy.  The
    label is which continuation occurred -- no symbolic state.  Zero (skip)
    when the model flag is off or the batch has no counterfactuals.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        e_exec = out.extras.get("energy_cf_exec")
        if e_exec is None:
            return out.step_states.sum() * 0.0
        e_alt = out.extras["energy_cf_alt"]  # [B, K]
        pair_loss = F.softplus(e_exec.unsqueeze(-1) - e_alt)
        return masked_mean(pair_loss, out.extras["energy_cf_valid"].float())


class EnergyPrefixRank(Objective):
    """Path-level (partial-trajectory) counterfactual ranking.

    The model emits an energy for every prefix of the observed imagined path
    and of each counterfactual path (the same actions with one infeasible
    intent inserted at a random depth).  Each path is reduced to ONE score,
    either the mean over its prefixes (``mean_prefix``, matching the planner's
    ``--aggregate mean_prefix``) or its endpoint (``endpoint``, the legacy
    behaviour that is blind to wasted steps and is kept only as an ablation).
    Loss = softplus(S_observed - S_counterfactual) over valid pairs: the path
    that actually occurred must score lower.  Label = which continuation
    occurred; no symbolic quantity is read.
    """

    def __init__(self, aggregate: str = "mean_prefix"):
        super().__init__()
        if aggregate not in {"mean_prefix", "endpoint"}:
            raise ValueError(f"unknown prefix aggregate: {aggregate}")
        self.aggregate = aggregate

    def _score(self, energies: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        v = valid.float()
        if self.aggregate == "mean_prefix":
            return (energies * v).sum(-1) / v.sum(-1).clamp(min=1.0)
        last = v.cumsum(-1).argmax(-1, keepdim=True)  # index of the final step
        return energies.gather(-1, last).squeeze(-1)

    def forward(self, out, batch: dict) -> torch.Tensor:
        e_obs = out.extras.get("energy_prefix_obs")
        if e_obs is None:
            return out.step_states.sum() * 0.0
        s_obs = self._score(e_obs, out.extras["energy_prefix_obs_valid"])
        s_cf = self._score(
            out.extras["energy_prefix_cf"], out.extras["energy_prefix_cf_valid"]
        )
        pair_valid = out.extras["energy_prefix_pair_valid"]
        pair_loss = F.softplus(s_obs.unsqueeze(-1) - s_cf)
        with torch.no_grad():
            ordered = (s_obs.unsqueeze(-1) < s_cf) & pair_valid
            out.extras["diag_energy_prefix_acc"] = (
                ordered.sum().float() / pair_valid.sum().clamp(min=1)
            )
            out.extras["diag_energy_prefix_pairs"] = pair_valid.sum().float()
        return masked_mean(pair_loss, pair_valid.float())
