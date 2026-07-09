"""Latent dynamics core shared by every JEPA track.

Owns the action-conditioned predictor, hierarchy (macro-actions +
high-level predictor in the same latent space, per HWM), Delta-JEPA
action decoder, and the value head. Tracks differ only in how they
encode states and actions; once those exist, the dynamics, losses and
planners are identical.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import MacroActionEncoder
from textjepa.models.delta_decoder import DeltaActionDecoder
from textjepa.models.heads import ValueHead
from textjepa.models.layers import mlp
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.predictor import ActionConditionedPredictor


class LatentDynamicsCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_action: int,
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        n_ops: int = 4,
        macro_k: int = 3,
        d_macro: int = 8,
        value_detach: bool = True,
    ):
        super().__init__()
        self.macro_k = macro_k
        self.value_detach = value_detach
        self.predictor = ActionConditionedPredictor(
            d_model, d_action, predictor_hidden_mult, predictor_layers
        )
        self.macro_encoder = MacroActionEncoder(d_action, d_macro)
        self.hi_predictor = ActionConditionedPredictor(
            d_model, d_macro, predictor_hidden_mult, predictor_layers
        )
        self.delta_decoder = DeltaActionDecoder(d_model, d_model, n_ops)
        self.value_head = ValueHead(d_model)
        # projects predicted states onto EMA chunk-embedding targets
        # (VL-JEPA-style continuous text-embedding prediction, no tokens)
        self.chunk_head = mlp([d_model, d_model * 2], d_model)

    def forward(
        self,
        s0: torch.Tensor,
        step_states: torch.Tensor,
        step_states_tgt: torch.Tensor,
        actions: torch.Tensor,
        action_emb_tgt: torch.Tensor,
        step_mask: torch.Tensor,
        step_emb_tgt: torch.Tensor | None = None,
    ) -> JEPAOutputs:
        prev_states = torch.cat([s0.unsqueeze(1), step_states[:, :-1]], dim=1)
        preds = self.predictor(prev_states, actions)
        rollout = self._rollout(s0, actions)
        op_logits, emb_pred = self.delta_decoder(step_states - prev_states)

        all_states = torch.cat([s0.unsqueeze(1), step_states], dim=1)
        value_in = all_states.detach() if self.value_detach else all_states
        value_pred = self.value_head(value_in, value_in[:, 0])

        hi_preds = hi_targets = hi_mask = None
        if self.macro_k and step_states.shape[1] >= self.macro_k:
            hi_preds, hi_targets, hi_mask = self._hierarchy(
                prev_states, actions, step_states_tgt, step_mask
            )

        extras = {}
        if step_emb_tgt is not None:
            extras["chunk_pred"] = self.chunk_head(preds)
            extras["chunk_pred_rollout"] = self.chunk_head(rollout)
            extras["step_emb_tgt"] = step_emb_tgt.detach()

        return JEPAOutputs(
            s0=s0,
            step_states=step_states,
            prev_states=prev_states,
            step_states_tgt=step_states_tgt.detach(),
            actions=actions,
            action_emb_tgt=action_emb_tgt.detach(),
            preds=preds,
            rollout=rollout,
            op_logits=op_logits,
            emb_pred=emb_pred,
            value_pred=value_pred,
            step_mask=step_mask,
            hi_preds=hi_preds,
            hi_targets=hi_targets,
            hi_mask=hi_mask,
            extras=extras,
        )

    def _rollout(self, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cur, out = s0, []
        for t in range(actions.shape[1]):
            cur = self.predictor(cur, actions[:, t])
            out.append(cur)
        return torch.stack(out, dim=1)

    def _hierarchy(
        self,
        prev_states: torch.Tensor,
        actions: torch.Tensor,
        step_states_tgt: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        K = self.macro_k
        B, T, d_a = actions.shape
        windows = actions.unfold(1, K, 1).permute(0, 1, 3, 2)  # [B, S, K, d_a]
        S = windows.shape[1]
        macro = self.macro_encoder(windows.reshape(B * S, K, d_a)).reshape(B, S, -1)
        hi_preds = self.hi_predictor(prev_states[:, :S], macro)
        return hi_preds, step_states_tgt[:, K - 1 :], step_mask[:, K - 1 :]
