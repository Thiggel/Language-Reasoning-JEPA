"""Action decoding from latent displacements."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective, latent_distance, masked_mean


class DeltaAction(Objective):
    """Legacy hybrid target: symbolic op class + EMA phrase embedding."""
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


class ObservedActionLDAD(Objective):
    """Faithful text LDAD: reconstruct the observed action phrase tokens.

    The target is external and fixed by the dataset, analogous to the raw
    continuous action in Delta-JEPA.  No operation class, learned action code,
    endpoint concatenation, or EMA action target is used.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        multistep = out.extras.get("observed_action_multistep_logits")
        if multistep is not None:
            horizon = multistep.shape[-3]
            n_starts = multistep.shape[1]
            if n_starts == 0:
                return out.step_states.sum() * 0.0
            target = torch.stack(
                [batch["action_tokens"][:, j : j + n_starts]
                 for j in range(horizon)],
                dim=2,
            )
            valid = torch.stack(
                [out.step_mask[:, j : j + n_starts] for j in range(horizon)],
                dim=2,
            )
            L = min(multistep.shape[-2], target.shape[-1])
            logits = multistep[..., :L, :]
            target = target[..., :L]
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1), reduction="none",
            ).reshape_as(target)
            mask = valid.unsqueeze(-1) & target.ne(0)
            return masked_mean(token_loss, mask.float())
        logits = out.extras.get("observed_action_logits")
        if logits is None:
            return out.step_states.sum() * 0.0
        target = out.extras.get("observed_action_targets", batch["action_tokens"])
        L = min(logits.shape[-2], target.shape[-1])
        logits = logits[..., :L, :]
        target = target[..., :L]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1), reduction="none",
        ).reshape_as(target)
        # PAD is id 0 in the shared synthetic vocabulary.  A transition must
        # be valid and a token position must contain observed action content.
        mask = out.step_mask.unsqueeze(-1) & (target != 0)
        return masked_mean(token_loss, mask.float())
