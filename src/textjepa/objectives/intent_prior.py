"""Generative intent-phrase prior: next-token CE on the flat trace stream."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from textjepa.objectives.base import Objective


class IntentPriorLM(Objective):
    """Next-token cross-entropy on the observed intent phrases (and, with
    ``lm_loss_on=all_solution``, the outcome tokens) of the flat stream.

    This is the menu-free PROPOSER of the flat intent JEPA: at plan time the
    same head samples intent phrases that the Energy then ranks.  It also
    replaces the old ``chunk_pred`` term (predicting frozen chunk embeddings
    from the state), which served as the anti-collusion anchor of the state
    space: here the anchor is the text itself.  Self-supervised (the targets
    are the observed tokens).  ``lm_mask`` marks target positions in the
    unshifted input sequence, exactly as in ``scripts/train_lm.py``.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        logits = out.extras.get("lm_logits")
        if logits is None:
            return out.step_states.sum() * 0.0
        tokens = out.extras["lm_tokens"]
        mask = out.extras["lm_mask"]
        pad_id = int(batch.get("pad_id", 0))
        logits = logits[:, :-1]
        tgt = tokens[:, 1:]
        m = mask[:, 1:] & (tgt != pad_id)
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
            reduction="none",
        ).reshape(tgt.shape)
        return (ce * m).sum() / m.sum().clamp(min=1)
