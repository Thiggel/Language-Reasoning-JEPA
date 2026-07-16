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
