"""Distributional representation-space losses for sentence-stream V-JEPA."""

from __future__ import annotations

import torch

from textjepa.objectives.base import Objective, masked_mean


class VariationalLatentPrediction(Objective):
    """Negative log likelihood of a sampled EMA target under the predictor."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "pred_logvar" not in out.extras:
            return out.preds.sum() * 0.0
        logvar = out.extras["pred_logvar"]
        target = out.extras["target_sample"]
        nll = 0.5 * (
            logvar + (target - out.preds).pow(2) * torch.exp(-logvar)
        ).mean(-1)
        return masked_mean(nll, out.step_mask.float())


class TargetDistributionKL(Objective):
    """KL(q_target(z|x) || N(0,I)), the V-JEPA target regularizer."""

    def __init__(self, free_nats: float = 0.0):
        super().__init__()
        self.free_nats = free_nats

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "target_kl" not in out.extras:
            return out.preds.sum() * 0.0
        kl = out.extras["target_kl"]
        if self.free_nats:
            kl = torch.clamp(kl - self.free_nats, min=0.0)
        return masked_mean(kl, out.step_mask.float())


class LatentDifferenceActionReconstruction(Objective):
    """Decode inferred action codes from online latent displacements.

    Delta-JEPA decodes an *observed* action.  With unobserved actions the only
    available target is the stop-gradient variational posterior code; this is a
    principled consistency constraint, but not by itself an identifiability
    guarantee.  Action-code variance is therefore audited separately.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "latent_ldad_pred" not in out.extras:
            return out.preds.sum() * 0.0
        err = (
            out.extras["latent_ldad_pred"] - out.extras["latent_ldad_tgt"]
        ).pow(2).mean(-1)
        return masked_mean(err, out.step_mask.float())


class CounterfactualVariationalPrediction(Objective):
    """Predict every rendered alternative outcome through its latent code."""

    def forward(self, out, batch: dict) -> torch.Tensor:
        nll = out.extras.get("counterfactual_variational_nll")
        if nll is None:
            return out.preds.sum() * 0.0
        return nll.mean()


class CounterfactualActionPrior(Objective):
    """Fit the plan-time prior to all same-state posterior outcome modes."""

    def __init__(self, free_nats: float = 0.0):
        super().__init__()
        self.free_nats = free_nats

    def forward(self, out, batch: dict) -> torch.Tensor:
        kl = out.extras.get("counterfactual_action_kl")
        if kl is None:
            return out.preds.sum() * 0.0
        if self.free_nats:
            kl = torch.clamp(kl - self.free_nats, min=0.0)
        return kl.mean()
