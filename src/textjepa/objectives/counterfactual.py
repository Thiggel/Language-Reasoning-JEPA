"""Objectives for observed counterfactual transitions without preferences."""

from __future__ import annotations

import torch

from textjepa.objectives.base import Objective, latent_distance, masked_mean


class CounterfactualOutcomePrediction(Objective):
    """Predict frozen embeddings of observed alternative next sentences.

    Unlike ActionRanking, this supplies no better/worse label and never touches
    the value head.  It is therefore the clean control for whether additional
    counterfactual transition coverage alone explains the ranking gains.
    """

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "cf_chunk_pred" not in out.extras:
            return out.preds.sum() * 0.0
        d = latent_distance(
            out.extras["cf_chunk_pred"], out.extras["cf_chunk_tgt"],
            self.kind, self.norm_targets,
        )
        return masked_mean(d, out.extras["cf_valid"].float())


class CounterfactualStatePrediction(Objective):
    """Match observed alternative actions to complete next-state latents.

    This is the counterfactual analogue of ``LatentPrediction``: the predictor
    receives the factual causal prefix plus an alternative action, and its
    output is regressed directly onto the EMA state obtained after the
    alternative's observed outcome.  It supplies no action-quality label.
    """

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "ga_cf_pred" not in out.extras:
            return out.preds.sum() * 0.0
        distance = latent_distance(
            out.extras["ga_cf_pred"],
            out.extras["ga_cf_target"],
            self.kind,
            self.norm_targets,
        )
        return masked_mean(
            distance, out.extras["ga_cf_valid"].float()
        )


class CounterfactualSlotPrediction(Objective):
    """Predict the exact changed-step anchor for each mechanical edit.

    This remains outcome supervision rather than preference supervision. Its
    purpose is to prevent a one-token effect from disappearing inside the
    global embedding of a long multi-step solution.
    """

    def __init__(self, kind: str = "smooth_l1", norm_targets: bool = True):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "cf_slot_pred" not in out.extras:
            return out.preds.sum() * 0.0
        distance = latent_distance(
            out.extras["cf_slot_pred"], out.extras["cf_slot_tgt"],
            self.kind, self.norm_targets,
        )
        return masked_mean(distance, out.extras["cf_slot_valid"].float())
