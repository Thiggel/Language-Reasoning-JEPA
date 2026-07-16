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

from textjepa.models.action import MacroActionModel
from textjepa.models.delta_decoder import DeltaActionDecoder
from textjepa.models.heads import (
    ActionSupportHead,
    ControllerOutcomeHead,
    SubgoalActionHead,
    MacroSupportHead,
    MacroValueHead,
    ValueHead,
)
from textjepa.models.layers import mlp
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.predictor import (
    ActionConditionedPredictor,
    CausalHistoryPredictor,
    CausalMacroPredictor,
    FiLMPredictor,
)


class LatentDynamicsCore(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_action: int,
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        predictor_heads: int = 8,
        n_ops: int = 4,
        macro_k: int = 3,
        d_macro: int = 8,
        value_detach: bool = True,
        geo_proj: bool = False,
        residual: bool = True,
        detach_targets: bool = True,
        predictor_kind: str = "causal",
        macro_encoder_kind: str = "transformer",
        macro_variational: bool = False,
        macro_concat_width: int = 8,
        high_predictor_kind: str = "causal",
        high_predictor_layers: int = 2,
        high_predictor_heads: int = 8,
        high_predictor_ff_mult: int = 4,
        high_predictor_residual: bool | None = None,
        dense_rollout_depth: int = 0,
        high_dense_rollout_depth: int = 0,
    ):
        super().__init__()
        self.d_action = d_action
        self.macro_k = macro_k
        self.value_detach = value_detach
        self.detach_targets = detach_targets
        self.dense_rollout_depth = max(0, int(dense_rollout_depth))
        self.high_dense_rollout_depth = max(
            0, int(high_dense_rollout_depth)
        )
        if predictor_kind == "causal":
            self.predictor = CausalHistoryPredictor(
                d_model,
                d_action,
                predictor_layers,
                predictor_heads,
                predictor_hidden_mult,
                max_steps=64,
                residual=residual,
            )
        elif predictor_kind in {"concat", "film"}:
            # Retained only to load and audit historical checkpoints. New
            # experiments use the causal predictor configured above.
            cls = (
                FiLMPredictor
                if predictor_kind == "film"
                else ActionConditionedPredictor
            )
            self.predictor = cls(
                d_model,
                d_action,
                predictor_hidden_mult,
                predictor_layers,
                residual,
            )
        else:
            raise ValueError(f"unknown predictor kind: {predictor_kind}")
        self.macro_encoder = MacroActionModel(
            d_action=d_action,
            d_state=d_model,
            d_macro=d_macro,
            span=macro_k,
            kind=macro_encoder_kind,
            variational=macro_variational,
            concat_width=macro_concat_width,
        )
        high_residual = (
            residual if high_predictor_residual is None
            else high_predictor_residual
        )
        if high_predictor_kind == "mlp":
            self.hi_predictor = ActionConditionedPredictor(
                d_model,
                d_macro,
                predictor_hidden_mult,
                high_predictor_layers,
                high_residual,
            )
        elif high_predictor_kind == "causal":
            self.hi_predictor = CausalMacroPredictor(
                d_model,
                d_macro,
                high_predictor_layers,
                high_predictor_heads,
                high_predictor_ff_mult,
                max_steps=64,
                residual=high_residual,
            )
        else:
            raise ValueError(
                f"unknown high predictor kind: {high_predictor_kind}"
            )
        self.delta_decoder = DeltaActionDecoder(d_model, d_model, n_ops)
        self.value_head = ValueHead(d_model)
        self.hi_value_head = ValueHead(d_model)
        self.macro_value_head = MacroValueHead(d_model, d_macro)
        self.macro_support_head = MacroSupportHead(d_model, d_macro)
        self.action_support_head = ActionSupportHead(d_model, d_action)
        self.subgoal_action_head = SubgoalActionHead(d_model, d_action)
        self.controller_remaining_head = ControllerOutcomeHead(d_model)
        self.controller_residual_head = ControllerOutcomeHead(d_model)
        # projects predicted states onto EMA chunk-embedding targets
        # (VL-JEPA-style continuous text-embedding prediction, no tokens)
        self.chunk_head = mlp([d_model, d_model * 2], d_model)
        # optional geometry projection: straightening/monotonicity act on
        # pi(s) instead of s, so the planning metric and the content
        # representation stop competing (Result 8 trade-off)
        self.geo_head = mlp([d_model, d_model], d_model) if geo_proj else None
        if self.macro_k <= 0:
            # Keep legacy-compatible module names in the checkpoint, but make
            # the flat paper model genuinely flat: no hierarchical parameter
            # is optimized or included in the active parameter count.
            for module in (
                self.macro_encoder,
                self.hi_predictor,
                self.hi_value_head,
                self.macro_value_head,
                self.macro_support_head,
                self.action_support_head,
                self.subgoal_action_head,
                self.controller_remaining_head,
                self.controller_residual_head,
            ):
                module.requires_grad_(False)

    def forward(
        self,
        s0: torch.Tensor,
        step_states: torch.Tensor,
        step_states_tgt: torch.Tensor,
        actions: torch.Tensor,
        action_emb_tgt: torch.Tensor,
        step_mask: torch.Tensor,
        step_emb_tgt: torch.Tensor | None = None,
        alt_actions: torch.Tensor | None = None,  # [B, T, K, d_action]
        preds_override: torch.Tensor | None = None,
        alt_preds_override: torch.Tensor | None = None,
    ) -> JEPAOutputs:
        prev_states = torch.cat([s0.unsqueeze(1), step_states[:, :-1]], dim=1)
        preds = (
            preds_override if preds_override is not None
            else self._predict_sequence(prev_states, actions, step_mask)
        )
        rollout = self._rollout(s0, actions)
        op_logits, emb_pred = self.delta_decoder(step_states - prev_states)

        all_states = torch.cat([s0.unsqueeze(1), step_states], dim=1)
        value_in = all_states.detach() if self.value_detach else all_states
        value_pred = self.value_head(value_in, value_in[:, 0])

        hi_preds = hi_targets = hi_mask = None
        hierarchy_extras = {}
        if self.macro_k and step_states.shape[1] >= self.macro_k:
            hi_preds, hi_targets, hi_mask, hierarchy_extras = self._hierarchy(
                prev_states, actions, step_states_tgt, step_mask
            )

        extras = dict(hierarchy_extras)
        if self.dense_rollout_depth:
            dense_predictions = [preds]
            cur = preds
            limit = min(self.dense_rollout_depth, actions.shape[1])
            for horizon in range(2, limit + 1):
                cur = self._predict_sequence(
                    cur[:, :-1],
                    actions[:, horizon - 1:],
                    step_mask[:, horizon - 1:],
                )
                dense_predictions.append(cur)
            extras["dense_rollout_predictions"] = tuple(dense_predictions)
            extras["dense_rollout_targets"] = tuple(
                step_states_tgt[:, horizon:]
                for horizon in range(limit)
            )
            extras["dense_rollout_masks"] = tuple(
                step_mask[:, horizon:]
                for horizon in range(limit)
            )
        if hi_preds is not None:
            hi_for_value = hi_preds.detach() if self.value_detach else hi_preds
            extras["hi_value_pred"] = self.hi_value_head(hi_for_value, s0)
            B = step_mask.shape[0]
            last = step_mask.sum(1).clamp(min=1) - 1
            goal = step_states_tgt[
                torch.arange(B, device=step_mask.device), last
            ]
            ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
            extras["hi_value_target"] = (
                ln(hi_targets) - ln(goal).unsqueeze(1)
            ).abs().mean(-1)
        if self.geo_head is not None:
            extras["geo_states"] = self.geo_head(all_states)
            extras["geo_states_tgt"] = self.geo_head(step_states_tgt.detach())
        if step_emb_tgt is not None:
            extras["chunk_pred"] = self.chunk_head(preds)
            extras["chunk_pred_rollout"] = self.chunk_head(rollout)
            extras["step_emb_tgt"] = step_emb_tgt.detach()
        if alt_actions is not None:
            # counterfactual candidates: predict + value-score the K
            # alternative actions from each visited state (ranking loss)
            B, T, K, d_a = alt_actions.shape
            prev_rep = prev_states.unsqueeze(2).expand(-1, -1, K, -1)
            alt_preds = (
                alt_preds_override if alt_preds_override is not None
                else self._predict_counterfactuals(
                    prev_states, actions, alt_actions, step_mask
                ).reshape(B, T * K, -1)
            )
            p_exec, p_alt = (
                (preds.detach(), alt_preds.detach())
                if self.value_detach
                else (preds, alt_preds)
            )
            extras["alt_preds"] = alt_preds.reshape(B, T, K, -1)
            extras["exec_value"] = self.value_head(p_exec, value_in[:, 0])
            extras["alt_value"] = self.value_head(
                p_alt, value_in[:, 0]
            ).reshape(B, T, K)
            if T > 1:
                # executed 2-step continuation F(F(s,a_t), a_{t+1}) — free
                # from the trace; lets CostRanking calibrate depth pricing
                if getattr(self.predictor, "causal_sequence", False):
                    preds2 = self._predict_second_steps(
                        prev_states, actions, preds, step_mask
                    )
                else:
                    preds2 = self.predictor(preds[:, :-1], actions[:, 1:])
                p2 = preds2.detach() if self.value_detach else preds2
                extras["exec2_value"] = self.value_head(p2, value_in[:, 0])

        return JEPAOutputs(
            s0=s0,
            step_states=step_states,
            prev_states=prev_states,
            step_states_tgt=(
                step_states_tgt.detach() if self.detach_targets else step_states_tgt
            ),
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
        if hasattr(self.predictor, "rollout"):
            return self.predictor.rollout(s0, actions)
        cur, out = s0, []
        for t in range(actions.shape[1]):
            cur = self.predictor(cur, actions[:, t])
            out.append(cur)
        return torch.stack(out, dim=1)

    def _predict_sequence(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self.predictor, "causal_sequence", False):
            return self.predictor(states, actions, valid)
        return self.predictor(states, actions)

    def _predict_counterfactuals(
        self,
        prev_states: torch.Tensor,
        actions: torch.Tensor,
        alt_actions: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict each alternative with its true causal prefix.

        Flattening ``T*K`` alternatives into one sequence leaks unrelated
        candidates into causal attention. This constructs an independent
        prefix ending at every ``(t, k)`` alternative instead.
        """
        B, T, K, d_action = alt_actions.shape
        d_state = prev_states.shape[-1]
        if not getattr(self.predictor, "causal_sequence", False):
            state = prev_states.unsqueeze(2).expand(-1, -1, K, -1)
            return self.predictor(
                state.reshape(B * T * K, d_state),
                alt_actions.reshape(B * T * K, d_action),
            ).reshape(B, T, K, d_state)
        states = prev_states[:, None, None].expand(-1, T, K, -1, -1)
        acts = actions[:, None, None].expand(-1, T, K, -1, -1).clone()
        anchor = torch.arange(T, device=actions.device)
        # Avoid advanced indexing here: indexing both trajectory axes with
        # ``anchor`` moves that axis in front of the batch axis and silently
        # changes the expected layout from [B,T,K,D] to [T,B,K,D].
        for step in range(T):
            acts[:, step, :, step] = alt_actions[:, step]
        prefix = torch.arange(T, device=actions.device)[None, :] <= anchor[:, None]
        valid = (
            step_mask[:, None, None, :]
            & prefix[None, :, None, :]
        ).expand(-1, -1, K, -1)
        flat_pred = self.predictor(
            states.reshape(B * T * K, T, d_state),
            acts.reshape(B * T * K, T, d_action),
            valid.reshape(B * T * K, T),
        )
        flat_pred = flat_pred.reshape(B, T, K, T, d_state)
        return torch.stack(
            [flat_pred[:, step, :, step] for step in range(T)], dim=1
        )

    def _predict_second_steps(
        self,
        prev_states: torch.Tensor,
        actions: torch.Tensor,
        first_predictions: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ``t+2`` from the observed prefix plus predicted ``t+1``."""
        B, T, d_state = prev_states.shape
        origins = T - 1
        states = prev_states[:, None].expand(-1, origins, -1, -1).clone()
        acts = actions[:, None].expand(-1, origins, -1, -1)
        anchor = torch.arange(origins, device=actions.device)
        states[:, anchor, anchor + 1] = first_predictions[:, anchor]
        prefix = (
            torch.arange(T, device=actions.device)[None, :]
            <= (anchor + 1)[:, None]
        )
        valid = step_mask[:, None, :] & prefix[None, :, :]
        predicted = self.predictor(
            states.reshape(B * origins, T, d_state),
            acts.reshape(B * origins, T, actions.shape[-1]),
            valid.reshape(B * origins, T),
        ).reshape(B, origins, T, d_state)
        return predicted[:, anchor, anchor + 1]

    def _hierarchy(
        self,
        prev_states: torch.Tensor,
        actions: torch.Tensor,
        step_states_tgt: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        K = self.macro_k
        B, T, d_a = actions.shape
        if getattr(self.hi_predictor, "causal_sequence", False):
            # The planner advances the high-level model by K low-level steps
            # per token.  Its causal training sequence must therefore use the
            # same non-overlapping stride; feeding overlapping windows here
            # would teach a one-step history and deploy it as a K-step history.
            start_list = list(range(0, T - K + 1, K))
            starts = torch.tensor(
                start_list, dtype=torch.long, device=actions.device
            )
            windows = torch.stack(
                [actions[:, t:t + K] for t in start_list], dim=1
            )
            macro_states = prev_states[:, starts]
            hi_targets = step_states_tgt[:, starts + K - 1]
            hi_mask = step_mask[:, starts + K - 1]
        else:
            windows = actions.unfold(1, K, 1).permute(
                0, 1, 3, 2
            )  # [B, S, K, d_a]
            macro_states = prev_states[:, :windows.shape[1]]
            hi_targets = step_states_tgt[:, K - 1:]
            hi_mask = step_mask[:, K - 1:]
        S = windows.shape[1]
        macro, extras = self.macro_encoder.training_code(
            windows.reshape(B * S, K, d_a),
            macro_states.reshape(B * S, -1),
        )
        macro = macro.reshape(B, S, -1)
        extras = {
            name: value.reshape(B, S, *value.shape[1:])
            for name, value in extras.items()
        }
        flat_windows = windows.reshape(B * S, K, d_a)
        low_endpoint = self._rollout(
            macro_states.reshape(B * S, -1), flat_windows
        )[:, -1]
        extras["hi_low_rollout_target"] = low_endpoint.reshape(B, S, -1)
        if getattr(self.hi_predictor, "causal_sequence", False):
            hi_preds = self.hi_predictor(
                macro_states, macro, hi_mask
            )
        else:
            hi_preds = self.hi_predictor(macro_states, macro)
        extras["macro_codes"] = macro
        if self.high_dense_rollout_depth:
            # Planning recursively feeds predicted macro states back into the
            # causal high-level model.  Train the exact same computation from
            # every observed macro origin, instead of only teacher-forcing a
            # full sequence of ground-truth waypoint states.
            dense_predictions = []
            dense_targets = []
            dense_masks = []
            limit = min(self.high_dense_rollout_depth, S)
            for horizon in range(1, limit + 1):
                n_origins = S - horizon + 1
                origins = macro_states[:, :n_origins]
                code_windows = torch.stack(
                    [macro[:, start:start + horizon]
                     for start in range(n_origins)],
                    dim=1,
                )
                rollout = self._high_rollout(
                    origins.reshape(B * n_origins, -1),
                    code_windows.reshape(
                        B * n_origins, horizon, macro.shape[-1]
                    ),
                )
                dense_predictions.append(
                    rollout[:, -1].reshape(B, n_origins, -1)
                )
                dense_targets.append(hi_targets[:, horizon - 1:])
                dense_masks.append(hi_mask[:, horizon - 1:])
            extras["high_dense_rollout_predictions"] = tuple(
                dense_predictions
            )
            extras["high_dense_rollout_targets"] = tuple(dense_targets)
            extras["high_dense_rollout_masks"] = tuple(dense_masks)
        return (
            hi_preds,
            hi_targets,
            hi_mask,
            extras,
        )

    def _high_rollout(
        self, start: torch.Tensor, codes: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self.hi_predictor, "rollout"):
            return self.hi_predictor.rollout(start, codes)
        cur = start
        predictions = []
        for step in range(codes.shape[1]):
            cur = self.hi_predictor(cur, codes[:, step])
            predictions.append(cur)
        return torch.stack(predictions, dim=1)
