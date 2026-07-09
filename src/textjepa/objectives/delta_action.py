"""Delta-JEPA action decoding from latent displacements (arXiv:2606.31232).

The op class and the EMA action-phrase embedding must be recoverable from
``s_{t+1} - s_t``: adjacent states cannot collapse without losing action
information, and different actions must displace the state differently.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, latent_distance, masked_mean


class DeltaAction(Objective):
    def __init__(self, ce_weight: float = 1.0, emb_weight: float = 1.0):
        super().__init__()
        self.ce_weight, self.emb_weight = ce_weight, emb_weight

    def forward(self, out, batch: dict) -> torch.Tensor:
        mask = out.step_mask.float()
        ce = F.cross_entropy(
            out.op_logits.transpose(1, 2), batch["op"], reduction="none"
        )
        emb = latent_distance(out.emb_pred, out.action_emb_tgt, "smooth_l1", True)
        return self.ce_weight * masked_mean(ce, mask) + self.emb_weight * masked_mean(
            emb, mask
        )
