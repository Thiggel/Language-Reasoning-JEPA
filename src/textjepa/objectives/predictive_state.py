"""Losses for cross-layer prediction, recurrent rollout, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from textjepa.models.action_transition import (
    activation_rms,
    parameter_free_rms_norm,
)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def masked_next_token_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and tokens must align before shifting")
    losses = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(target_mask)
    return masked_mean(losses, target_mask)


@dataclass
class TransitionLoss:
    total: torch.Tensor
    cosine: torch.Tensor
    scale: torch.Tensor
    normalized_mse: torch.Tensor
    mean_cosine: torch.Tensor
    predicted_rms: torch.Tensor
    target_rms: torch.Tensor


def transition_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale_weight: float = 0.01,
    eps: float = 1e-6,
) -> TransitionLoss:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    normalized_prediction = parameter_free_rms_norm(prediction, eps)
    normalized_target = parameter_free_rms_norm(target.detach(), eps)
    cos = F.cosine_similarity(
        normalized_prediction.float(), normalized_target.float(), dim=-1
    )
    cosine_loss = masked_mean(1.0 - cos, mask)
    pred_rms = activation_rms(prediction, eps)
    target_rms = activation_rms(target.detach(), eps)
    scale = F.huber_loss(
        pred_rms.log(), target_rms.log(), reduction="none"
    )
    scale_loss = masked_mean(scale, mask)
    normalized_mse = masked_mean(
        (normalized_prediction.float() - normalized_target.float())
        .square().mean(dim=-1),
        mask,
    )
    return TransitionLoss(
        total=cosine_loss + scale_weight * scale_loss,
        cosine=cosine_loss,
        scale=scale_loss,
        normalized_mse=normalized_mse,
        mean_cosine=masked_mean(cos, mask),
        predicted_rms=masked_mean(pred_rms, mask),
        target_rms=masked_mean(target_rms, mask),
    )


@dataclass
class Stage1Loss:
    total: torch.Tensor
    ntp: torch.Tensor
    transition: TransitionLoss | None


def stage1_loss(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    prediction: torch.Tensor | None,
    target_state: torch.Tensor | None,
    prediction_weight: float,
    scale_weight: float,
) -> Stage1Loss:
    ntp = masked_next_token_loss(logits, input_ids, target_mask)
    if prediction is None:
        return Stage1Loss(total=ntp, ntp=ntp, transition=None)
    if target_state is None:
        raise ValueError("a transition prediction needs a target")
    auxiliary = transition_loss(
        prediction, target_state, target_mask, scale_weight=scale_weight
    )
    return Stage1Loss(
        total=ntp + prediction_weight * auxiliary.total,
        ntp=ntp,
        transition=auxiliary,
    )


@dataclass
class RolloutStepLoss:
    total: torch.Tensor
    state: torch.Tensor
    kl: torch.Tensor
    cross_entropy: torch.Tensor
    teacher_cross_entropy: torch.Tensor
    excess_nll: torch.Tensor
    top1_agreement: torch.Tensor
    top20_agreement: torch.Tensor
    log_rms_drift: torch.Tensor


def rollout_step_loss(
    *,
    predicted_state: torch.Tensor,
    target_state: torch.Tensor,
    jump_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    next_token: torch.Tensor,
    mask: torch.Tensor | None = None,
    ce_weight: float = 0.1,
) -> RolloutStepLoss:
    if mask is None:
        mask = torch.ones(
            predicted_state.shape[:-1], dtype=torch.bool,
            device=predicted_state.device,
        )
    state = transition_loss(
        predicted_state, target_state, mask, scale_weight=0.0
    ).cosine
    teacher_log_prob = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    jump_log_prob = F.log_softmax(jump_logits.float(), dim=-1)
    teacher_prob = teacher_log_prob.exp()
    kl_each = (teacher_prob * (teacher_log_prob - jump_log_prob)).sum(-1)
    kl = masked_mean(kl_each, mask)
    ce_each = F.cross_entropy(
        jump_logits.float().reshape(-1, jump_logits.shape[-1]),
        next_token.reshape(-1), reduction="none",
    ).reshape_as(mask)
    ce = masked_mean(ce_each, mask)
    teacher_ce_each = F.cross_entropy(
        teacher_logits.float().reshape(-1, teacher_logits.shape[-1]),
        next_token.reshape(-1), reduction="none",
    ).reshape_as(mask)
    teacher_ce = masked_mean(teacher_ce_each, mask)
    agreement = masked_mean(
        (jump_logits.argmax(-1) == teacher_logits.argmax(-1)).float(), mask
    )
    teacher_top = teacher_logits.argmax(-1, keepdim=True)
    jump_top20 = jump_logits.topk(min(20, jump_logits.shape[-1]), dim=-1).indices
    top20 = masked_mean((jump_top20 == teacher_top).any(-1).float(), mask)
    drift = masked_mean(
        (activation_rms(predicted_state).log()
         - activation_rms(target_state.detach()).log()).abs(),
        mask,
    )
    return RolloutStepLoss(
        total=state + kl + ce_weight * ce,
        state=state,
        kl=kl,
        cross_entropy=ce,
        teacher_cross_entropy=teacher_ce,
        excess_nll=ce - teacher_ce,
        top1_agreement=agreement,
        top20_agreement=top20,
        log_rms_drift=drift,
    )


def normalized_discount_weights(
    horizon: int, discount: float = 0.97, *, device=None
) -> torch.Tensor:
    if horizon < 1 or not 0.0 < discount <= 1.0:
        raise ValueError("invalid horizon or discount")
    weights = discount ** torch.arange(horizon, device=device).float()
    return weights / weights.sum()
