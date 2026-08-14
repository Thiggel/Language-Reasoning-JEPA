"""Action decoding from latent displacements, and open-ended proposal."""

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


class ObservedActionLDADPredictorCycle(Objective):
    """Predictor-cycle LDAD: decode the observed phrase from the IMAGINED
    displacement predictor(s_t, u_t) - s_t (the exact eval-time ldad_cycle
    input, which the standard LDAD loss never trains on).  Same token CE,
    targets, and masking as :class:`ObservedActionLDAD`; gradients flow into
    the predictor by design.  Zero when the model was built without
    ``observed_action_ldad_predictor_cycle``.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        logits = out.extras.get("observed_action_pred_cycle_logits")
        if logits is None:
            return out.step_states.sum() * 0.0
        target = batch["action_tokens"]
        L = min(logits.shape[-2], target.shape[-1])
        logits = logits[..., :L, :]
        target = target[..., :L]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1), reduction="none",
        ).reshape_as(target)
        mask = out.step_mask.unsqueeze(-1) & (target != 0)
        return masked_mean(token_loss, mask.float())


def _mean_phrase_log_prob(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean token log-prob of each candidate's own phrase (PAD masked).

    Trailing dims of ``logits`` are [.., max_len, vocab]; ``target`` is
    [.., L].  Returns [..] — one cycle score per candidate, the training-time
    counterpart of ``planning/ldad_decode.phrase_log_probs``.
    """
    L = min(logits.shape[-2], target.shape[-1])
    log_probs = logits[..., :L, :].log_softmax(-1)
    target = target[..., :L]
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    content = (target != 0).float()
    return (token_lp * content).sum(-1) / content.sum(-1).clamp(min=1.0)


class ObservedActionLDADCFContrast(Objective):
    """Rank the observed action's LDAD cycle score above each counterfactual
    candidate's (logistic pairwise, like the geo ranking losses).  A
    candidate's cycle score is the mean token log-prob of its OWN phrase under
    the decoder applied to its predictor displacement.  Self-supervised:
    candidates are the batch's existing ranking alternatives; the label is
    which continuation actually occurred.  Zero (skip) when the batch carries
    no counterfactual fields or the model flag is off.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        exec_logits = out.extras.get("ldad_cf_exec_logits")
        if exec_logits is None:
            return out.step_states.sum() * 0.0
        score_exec = _mean_phrase_log_prob(
            exec_logits, out.extras["ldad_cf_exec_targets"]
        )  # [B]
        score_alt = _mean_phrase_log_prob(
            out.extras["ldad_cf_alt_logits"],
            out.extras["ldad_cf_alt_targets"],
        )  # [B, K]
        # -log sigmoid(observed - counterfactual) per valid pair.
        pair_loss = F.softplus(score_alt - score_exec.unsqueeze(-1))
        return masked_mean(pair_loss, out.extras["ldad_cf_valid"].float())


class ActionGeneration(Objective):
    """Teacher-forced CE of the observed intent phrase given the state.

    Trains the state-conditioned generator head used for open-ended
    (catalogue-free) proposals.  The model detaches the state before this
    head, so the loss trains the head alone and cannot shape the JEPA
    representation.  Masking is the LDAD mask (a valid transition, a non-PAD
    token position) PLUS the first PAD position of each phrase.  That extra
    position is the end marker: sampling stops at the first PAD, so leaving it
    unsupervised (as the LDAD scoring loss does, since it never generates)
    would make it impossible for the head to terminate a phrase — every
    proposal would run to ``max_len`` and fail to parse.
    """

    def forward(self, out, batch: dict) -> torch.Tensor:
        logits = out.extras.get("action_generator_logits")
        if logits is None:
            return out.step_states.sum() * 0.0
        target = batch["action_tokens"]
        L = min(logits.shape[-2], target.shape[-1])
        logits = logits[..., :L, :]
        target = target[..., :L]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1), reduction="none",
        ).reshape_as(target)
        content = target != 0
        # First PAD after the phrase: the end marker the sampler stops at.
        end_marker = torch.arange(
            L, device=target.device
        ).expand_as(target) == content.sum(-1, keepdim=True)
        mask = out.extras["action_generator_valid"].unsqueeze(-1) & (
            content | end_marker
        )
        return masked_mean(token_loss, mask.float())
