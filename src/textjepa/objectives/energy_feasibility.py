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
