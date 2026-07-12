"""Next-chunk embedding prediction (VL-JEPA-style continuous targets).

Predicts the (frozen or EMA) chunk-encoder embedding of the next step from
(s_t, a_t) — still reconstruction-free, but anchors outcome content (e.g.
computed values) that pure state-prediction lets the encoder smooth away:
with a frozen random-init anchor the targets are fixed and provably retain
surface information, so encoder/predictor collusion cannot erase it.
"""

from __future__ import annotations

import torch

from textjepa.objectives.base import Objective, latent_distance, masked_mean


class ChunkPrediction(Objective):
    def __init__(
        self,
        kind: str = "smooth_l1",
        norm_targets: bool = True,
        rollout_weight: float = 0.5,
    ):
        super().__init__()
        self.kind = kind
        self.norm_targets = norm_targets
        self.rollout_weight = rollout_weight

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "chunk_pred" not in out.extras:
            return torch.zeros((), device=out.preds.device)
        tgt = out.extras["step_emb_tgt"]
        mask = out.step_mask.float()
        loss = masked_mean(
            latent_distance(out.extras["chunk_pred"], tgt, self.kind, self.norm_targets),
            mask,
        )
        if self.rollout_weight > 0:
            loss = loss + self.rollout_weight * masked_mean(
                latent_distance(
                    out.extras["chunk_pred_rollout"], tgt, self.kind, self.norm_targets
                ),
                mask,
            )
        return loss


class SlotAnchor(Objective):
    """Per-slot outcome anchor (edit track): predict the frozen anchor
    embedding of the CHANGED sentence at the edited position from
    F(s, a) — targets at the granularity the action operates on."""

    def __init__(self, kind: str = "smooth_l1"):
        super().__init__()
        self.kind = kind

    def forward(self, out, batch: dict) -> torch.Tensor:
        if "slot_pred" not in out.extras:
            return out.step_states.sum() * 0.0
        d = latent_distance(
            out.extras["slot_pred"], out.extras["slot_tgt"], self.kind, True
        )
        mask = (batch["changed_valid"] & out.step_mask).float()
        return masked_mean(d, mask)
