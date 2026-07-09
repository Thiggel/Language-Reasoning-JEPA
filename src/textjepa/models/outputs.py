"""Shared output container for all JEPA tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class JEPAOutputs:
    s0: torch.Tensor  # [B, D] state before any step/edit
    step_states: torch.Tensor  # [B, T, D] encoded state after step t
    prev_states: torch.Tensor  # [B, T, D] state before step t
    step_states_tgt: torch.Tensor  # [B, T, D] EMA targets, detached
    actions: torch.Tensor  # [B, T, d_a] bottlenecked action codes
    action_emb_tgt: torch.Tensor  # [B, T, D] EMA phrase embeddings, detached
    preds: torch.Tensor  # [B, T, D] teacher-forced F(s_t, a_t)
    rollout: torch.Tensor  # [B, T, D] open-loop rollout from s0
    op_logits: torch.Tensor  # [B, T, n_ops] LDAD op decoding from delta
    emb_pred: torch.Tensor  # [B, T, D] LDAD phrase-embedding decoding
    value_pred: torch.Tensor  # [B, T+1] predicted remaining steps
    step_mask: torch.Tensor  # [B, T] bool
    hi_preds: torch.Tensor | None = None  # [B, S, D] macro-action predictions
    hi_targets: torch.Tensor | None = None  # [B, S, D]
    hi_mask: torch.Tensor | None = None  # [B, S] bool
    extras: dict = field(default_factory=dict)
