"""Discourse-JEPA: an action-conditioned latent world model over reasoning steps.

World state  = compressed discourse state s_t (what has been established).
Action       = tiny latent code of an intent phrase ("derive X from A plus B").
Transition   = predictor F(s_t, a_t) -> s_{t+1}, trained against EMA targets.
Consequences (e.g. the arithmetic result stated in the next step) are never
given to the predictor — it must model them in latent space.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import (
    ActionEncoder,
    TokenBottleneckActionEncoder,
    VariationalAction,
)
from textjepa.models.core import LatentDynamicsCore
from textjepa.models.ema import EMATeacher
from textjepa.models.layers import TokenTransformer
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.state_model import DiscourseStateModel

DiscourseOutputs = JEPAOutputs  # backwards-compatible alias


class DiscourseJEPA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        chunk_layers: int = 2,
        chunk_heads: int = 4,
        state_layers: int = 4,
        state_heads: int = 8,
        ff_mult: int = 4,
        max_chunk_len: int = 48,
        max_chunks: int = 64,
        d_action: int = 16,
        fsq_levels: list[int] | None = None,
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        predictor_heads: int = 8,
        n_ops: int = 4,
        macro_k: int = 3,
        d_macro: int = 8,
        value_detach: bool = True,
        dropout: float = 0.0,
        chunk_target: str = "frozen",  # "frozen" | "ema" anchor for chunk_pred
        freeze_encoders: bool = False,  # baseline: random frozen representation
        geo_proj: bool = False,  # geometry losses act on a learned projection
        # "ema" | "online" (sg online states) | "online_nosg" (no stopgrad;
        # stability ablation for this repository's hybrid displacement loss)
        state_target: str = "ema",
        predictor_residual: bool = True,
        predictor_kind: str = "causal",
        action_encoder_kind: str = "pooled",
        action_token_dim: int = 8,
        variational_actions: bool = False,
        observed_action_ldad: bool = False,
        observed_action_ldad_horizon: int = 1,
        ldad_decoder_layers: int = 2,
        macro_encoder_kind: str = "transformer",
        macro_variational: bool = False,
        macro_concat_width: int = 8,
        high_predictor_kind: str = "causal",
        high_predictor_layers: int = 2,
        high_predictor_heads: int = 8,
        high_predictor_ff_mult: int = 4,
        high_predictor_residual: bool | None = None,
        action_support_states: str = "true",
        macro_support_scales: list[float] | None = None,
        dense_rollout_depth: int = 0,
        high_dense_rollout_depth: int = 0,
    ):
        super().__init__()
        self.chunk_target = chunk_target
        self.freeze_encoders = freeze_encoders
        self.state_target = state_target
        if action_support_states not in {"true", "all"}:
            raise ValueError(
                f"unknown action-support state mode: {action_support_states}"
            )
        self.action_support_states = action_support_states
        self.macro_support_scales = tuple(macro_support_scales or [3.0])
        self.chunk_encoder = TokenTransformer(
            vocab_size, pad_id, d_model, chunk_layers, chunk_heads,
            ff_mult, max_chunk_len, dropout,
        )
        self.state_model = DiscourseStateModel(
            d_model, state_layers, state_heads, ff_mult, max_chunks, dropout
        )
        self.action_encoder_kind = action_encoder_kind
        if action_encoder_kind == "pooled":
            self.action_encoder = ActionEncoder(
                d_model, d_action, fsq_levels=fsq_levels
            )
        elif action_encoder_kind == "token_bottleneck":
            self.action_encoder = TokenBottleneckActionEncoder(
                d_model, d_action, max_chunk_len, action_token_dim, fsq_levels
            )
        else:
            raise ValueError(f"unknown action_encoder_kind: {action_encoder_kind}")
        self.var_action = (
            VariationalAction(d_model, d_action) if variational_actions else None
        )
        from textjepa.models.delta_decoder import (
            MultiStepObservedActionDecoder,
            ObservedActionDecoder,
        )

        self.observed_action_ldad_horizon = int(observed_action_ldad_horizon)
        decoder_cls = (
            ObservedActionDecoder
            if self.observed_action_ldad_horizon == 1
            else MultiStepObservedActionDecoder
        )
        decoder_kwargs = (
            {}
            if self.observed_action_ldad_horizon == 1
            else {"horizon": self.observed_action_ldad_horizon}
        )
        self.observed_action_decoder = (
            decoder_cls(
                d_model, vocab_size, max_chunk_len,
                n_layers=ldad_decoder_layers, n_heads=chunk_heads,
                **decoder_kwargs,
            )
            if observed_action_ldad else None
        )
        from textjepa.models.layers import mlp as _mlp

        self.act_decode = (
            _mlp([d_action, d_model], d_model) if variational_actions else None
        )
        self.core = LatentDynamicsCore(
            d_model=d_model,
            d_action=d_action,
            predictor_hidden_mult=predictor_hidden_mult,
            predictor_layers=predictor_layers,
            predictor_heads=predictor_heads,
            n_ops=n_ops,
            macro_k=macro_k,
            d_macro=d_macro,
            value_detach=value_detach,
            geo_proj=geo_proj,
            residual=predictor_residual,
            detach_targets=state_target != "online_nosg",
            predictor_kind=predictor_kind,
            macro_encoder_kind=macro_encoder_kind,
            macro_variational=macro_variational,
            macro_concat_width=macro_concat_width,
            high_predictor_kind=high_predictor_kind,
            high_predictor_layers=high_predictor_layers,
            high_predictor_heads=high_predictor_heads,
            high_predictor_ff_mult=high_predictor_ff_mult,
            high_predictor_residual=high_predictor_residual,
            dense_rollout_depth=dense_rollout_depth,
            high_dense_rollout_depth=high_dense_rollout_depth,
        )
        self.chunk_teacher = EMATeacher(self.chunk_encoder)
        self.state_teacher = EMATeacher(self.state_model)
        # frozen random-init copy: fixed, informative chunk-embedding targets
        # (never updated; random features provably retain surface content)
        self.chunk_anchor = EMATeacher(self.chunk_encoder)
        if freeze_encoders:
            self.chunk_encoder.requires_grad_(False)
            self.state_model.requires_grad_(False)

    # convenience handles used by planners
    @property
    def predictor(self):
        return self.core.predictor

    @property
    def value_head(self):
        return self.core.value_head

    # ------------------------------------------------------------------ #
    # encoding helpers (also used by the planner and probing suite)
    # ------------------------------------------------------------------ #
    def encode_chunks(
        self, tokens: torch.Tensor, teacher: bool = False
    ) -> torch.Tensor:
        """[B, C, L] token ids -> [B, C, D] chunk embeddings."""
        B, C, L = tokens.shape
        enc = self.chunk_teacher if teacher else self.chunk_encoder
        return enc(tokens.reshape(B * C, L)).reshape(B, C, -1)

    def encode_states(
        self,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        step_tokens: torch.Tensor,
        step_mask: torch.Tensor,
        teacher: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_emb = self.encode_chunks(prompt_tokens, teacher)
        step_emb = self.encode_chunks(step_tokens, teacher)
        model = self.state_teacher if teacher else self.state_model
        return model(prompt_emb, prompt_mask, step_emb, step_mask)

    def encode_actions(self, action_tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, L] action-phrase tokens -> [B, T, d_action]."""
        if self.action_encoder_kind == "pooled":
            return self.action_encoder(self.encode_chunks(action_tokens))
        token_emb = self.chunk_encoder.tok(action_tokens)
        return self.action_encoder(
            token_emb, action_tokens.ne(self.chunk_encoder.pad_id)
        )

    def _encode_alt(self, batch: dict) -> torch.Tensor | None:
        """[B, T, K, L] alternative-action tokens -> [B, T, K, d_action]."""
        if "alt_tokens" not in batch:
            return None
        B, T, K, L = batch["alt_tokens"].shape
        return self.encode_actions(
            batch["alt_tokens"].reshape(B, T * K, L)
        ).reshape(B, T, K, -1)

    @torch.no_grad()
    def update_teachers(self, momentum: float) -> None:
        self.chunk_teacher.update(self.chunk_encoder, momentum)
        self.state_teacher.update(self.state_model, momentum)

    # ------------------------------------------------------------------ #
    def forward(self, batch: dict) -> JEPAOutputs:
        s0, step_states = self.encode_states(
            batch["prompt_tokens"], batch["prompt_mask"],
            batch["step_tokens"], batch["step_mask"],
        )
        if self.state_target == "ema":
            with torch.no_grad():
                _, step_states_tgt = self.encode_states(
                    batch["prompt_tokens"], batch["prompt_mask"],
                    batch["step_tokens"], batch["step_mask"], teacher=True,
                )
        elif self.state_target == "online":
            step_states_tgt = step_states.detach()
        else:  # online_nosg: gradients flow through the target side too
            step_states_tgt = step_states
        with torch.no_grad():
            action_emb_tgt = self.encode_chunks(batch["action_tokens"], teacher=True)
            if self.chunk_target == "frozen":
                B, C, L = batch["step_tokens"].shape
                step_emb_tgt = self.chunk_anchor(
                    batch["step_tokens"].reshape(B * C, L)
                ).reshape(B, C, -1)
            else:
                step_emb_tgt = self.encode_chunks(batch["step_tokens"], teacher=True)

        var_extras = {}
        if self.var_action is not None:
            prev = torch.cat([s0.unsqueeze(1), step_states[:, :-1]], dim=1)
            actions, q_params = self.var_action.sample_posterior(
                prev, step_states.detach()
            )
            p_params = self.var_action.prior_params(prev.detach())
            var_extras["action_kl"] = self.var_action.kl(q_params, p_params)
            # detached readout: code -> intent anchor embedding (no leakage)
            with torch.no_grad():
                B, C, L = batch["action_tokens"].shape
                intent_anchor = self.chunk_anchor(
                    batch["action_tokens"].reshape(B * C, L)
                ).reshape(B, C, -1)
            var_extras["act_decode"] = self.act_decode(actions.detach())
            var_extras["act_decode_tgt"] = intent_anchor
        else:
            actions = self.encode_actions(batch["action_tokens"])
        alt_actions = self._encode_alt(batch)
        out = self.core(
            s0, step_states, step_states_tgt, actions, action_emb_tgt,
            batch["step_mask"], step_emb_tgt=step_emb_tgt,
            alt_actions=alt_actions,
        )
        if out.hi_preds is not None:
            K = self.core.macro_k
            if getattr(self.core.hi_predictor, "causal_sequence", False):
                hi_remaining = batch["remaining"][:, K - 1::K]
            else:
                hi_remaining = batch["remaining"][:, K - 1:]
            hi_remaining = hi_remaining[:, :out.hi_preds.shape[1]].float()
            out.extras["hi_remaining_target"] = hi_remaining
            dense_predictions = out.extras.get(
                "high_dense_rollout_predictions"
            )
            if dense_predictions:
                dense_values = []
                dense_targets = []
                for horizon, prediction in enumerate(
                    dense_predictions, start=1
                ):
                    dense_values.append(self.core.hi_value_head(
                        prediction,
                        s0.unsqueeze(1).expand(
                            -1, prediction.shape[1], -1
                        ),
                    ))
                    dense_targets.append(hi_remaining[:, horizon - 1:])
                out.extras["high_dense_value_predictions"] = tuple(
                    dense_values
                )
                out.extras["high_dense_value_targets"] = tuple(
                    dense_targets
                )
        out.extras.update(var_extras)
        if "macro_alt_action_tokens" in batch:
            self._macro_counterfactuals(batch, out)
        if "action_candidate_tokens" in batch:
            self._action_support(batch, out)
        if self.observed_action_decoder is not None:
            if self.observed_action_ldad_horizon == 1:
                out.extras["observed_action_logits"] = self.observed_action_decoder(
                    out.step_states - out.prev_states
                )
            else:
                horizon = self.observed_action_ldad_horizon
                n_starts = max(out.step_states.shape[1] - horizon + 1, 0)
                displacement = (
                    out.step_states[:, horizon - 1 : horizon - 1 + n_starts]
                    - out.prev_states[:, :n_starts]
                )
                out.extras["observed_action_multistep_logits"] = (
                    self.observed_action_decoder(displacement)
                )
        if "alt_preds" in out.extras and "alt_step_tokens" in batch:
            B, T, K, L = batch["alt_step_tokens"].shape
            with torch.no_grad():
                alt_tgt = self.chunk_anchor(
                    batch["alt_step_tokens"].reshape(B * T * K, L)
                ).reshape(B, T, K, -1)
            out.extras["cf_chunk_pred"] = self.core.chunk_head(
                out.extras["alt_preds"]
            )
            out.extras["cf_chunk_tgt"] = alt_tgt
            out.extras["cf_valid"] = batch["alt_remaining"] >= 0
        if "ga_t" in batch and (batch["ga_t"] >= 0).any():
            self._geo_rank(batch, out)
        return out

    def _action_support(self, batch: dict, out) -> None:
        """Score every problem action at every observed prefix state."""
        actions = self.encode_actions(batch["action_candidate_tokens"])
        B, T, d_state = out.prev_states.shape
        V = actions.shape[1]
        states = [out.prev_states]
        if self.action_support_states == "all":
            states.extend([
                torch.cat([out.s0.unsqueeze(1), out.preds[:, :-1]], dim=1),
                torch.cat([out.s0.unsqueeze(1), out.rollout[:, :-1]], dim=1),
            ])
        states = torch.stack(states, dim=1)
        if self.core.value_detach:
            states = states.detach()
        action_features = actions.detach() if self.core.value_detach else actions
        M = states.shape[1]
        logits = self.core.action_support_head(
            states.unsqueeze(3).expand(B, M, T, V, d_state),
            action_features.unsqueeze(1).unsqueeze(1).expand(
                B, M, T, V, -1
            ),
        )
        valid = (
            out.step_mask.unsqueeze(-1)
            & batch["action_candidate_mask"].unsqueeze(1)
        ).unsqueeze(1).expand(B, M, T, V)
        target = batch["action_feasible"].unsqueeze(1).expand(B, M, T, V)
        if M == 1:
            logits = logits[:, 0]
            valid = valid[:, 0]
            target = target[:, 0]
        out.extras.update(
            action_support_logits=logits,
            action_support_valid=valid,
            action_support_target=target,
        )

    def _macro_counterfactuals(self, batch: dict, out) -> None:
        """Encode valid alternative macro chunks and their true outcomes."""
        tokens = batch["macro_alt_action_tokens"]
        B, A, K, L = tokens.shape
        low_actions = self.encode_actions(
            tokens.reshape(B * A, K, L)
        )
        macro = self.core.macro_encoder(low_actions).reshape(B, A, -1)
        anchor_idx = batch["macro_alt_t"].clamp_min(0)
        anchor = out.prev_states[
            torch.arange(B, device=anchor_idx.device), anchor_idx
        ]
        anchor = anchor.unsqueeze(1).expand(-1, A, -1)
        initial = out.s0.unsqueeze(1).expand(-1, A, -1)
        flat_anchor = anchor.reshape(B * A, -1)
        flat_initial = initial.reshape(B * A, -1)
        flat_macro = macro.reshape(B * A, -1)
        pred = self.core.hi_predictor(flat_anchor, flat_macro)
        low_target = flat_anchor
        for step in range(K):
            low_target = self.core.predictor(low_target, low_actions[:, step])

        step_tokens = batch["macro_alt_step_tokens"]
        step_mask = batch["macro_alt_step_mask"]
        _, _, T, Ls = step_tokens.shape
        prompt = batch["prompt_tokens"].unsqueeze(1).expand(-1, A, -1, -1)
        prompt_mask = batch["prompt_mask"].unsqueeze(1).expand(-1, A, -1)
        with torch.no_grad():
            _, target_states = self.encode_states(
                prompt.reshape(B * A, prompt.shape[-2], prompt.shape[-1]),
                prompt_mask.reshape(B * A, prompt_mask.shape[-1]),
                step_tokens.reshape(B * A, T, Ls),
                step_mask.reshape(B * A, T),
                teacher=True,
            )
            flat_mask = step_mask.reshape(B * A, T)
            last = flat_mask.sum(1).clamp_min(1) - 1
            target = target_states[
                torch.arange(B * A, device=last.device), last
            ]
        value_detach = self.core.value_detach
        value_pred = pred.detach() if value_detach else pred
        value_anchor = flat_anchor.detach() if value_detach else flat_anchor
        value_initial = flat_initial.detach() if value_detach else flat_initial
        value_macro = flat_macro.detach() if value_detach else flat_macro
        cf_target = target.reshape(B, A, -1)
        first_actions = low_actions[:, 0].reshape(B, A, -1)
        subgoal_action_cost = self.core.subgoal_action_head(
            value_anchor.reshape(B, A, -1).unsqueeze(2).expand(
                B, A, A, -1
            ),
            cf_target.detach().unsqueeze(2).expand(B, A, A, -1),
            first_actions.detach().unsqueeze(1).expand(B, A, A, -1),
        )
        subgoal_action_cost_pred = self.core.subgoal_action_head(
            value_anchor.reshape(B, A, -1).unsqueeze(2).expand(
                B, A, A, -1
            ),
            pred.detach().reshape(B, A, -1).unsqueeze(2).expand(
                B, A, A, -1
            ),
            first_actions.detach().unsqueeze(1).expand(B, A, A, -1),
        )
        first_tokens = tokens[:, :, 0]
        subgoal_action_positive = (
            tokens.unsqueeze(2)
            == first_tokens.unsqueeze(1).unsqueeze(3)
        ).all(-1).any(-1)
        state_value = self.core.hi_value_head(value_pred, value_initial)
        action_value = self.core.macro_value_head(
            value_anchor, value_initial, value_macro
        )
        support_pos = self.core.macro_support_head(value_anchor, value_macro)
        shuffled = torch.roll(value_macro, shifts=1, dims=0)
        support_shuffled = self.core.macro_support_head(value_anchor, shuffled)
        scale = macro.detach().std(dim=(0, 1), unbiased=False).clamp_min(0.1)
        factors = value_macro.new_tensor(self.macro_support_scales)
        perturbed = value_macro.unsqueeze(1) + (
            torch.randn(
                value_macro.shape[0], factors.numel(), value_macro.shape[1],
                device=value_macro.device,
                dtype=value_macro.dtype,
            )
            * scale.view(1, 1, -1)
            * factors.view(1, -1, 1)
        )
        support_perturbed = self.core.macro_support_head(
            value_anchor.unsqueeze(1).expand(-1, factors.numel(), -1),
            perturbed,
        )
        perturbed_value = self.core.macro_value_head(
            value_anchor.unsqueeze(1).expand(-1, factors.numel(), -1),
            value_initial.unsqueeze(1).expand(-1, factors.numel(), -1),
            perturbed,
        )
        out.extras.update(
            macro_cf_pred=pred.reshape(B, A, -1),
            macro_cf_target=cf_target,
            macro_cf_low_target=low_target.reshape(B, A, -1),
            macro_cf_codes=macro,
            macro_cf_anchor=anchor,
            macro_cf_state_value=state_value.reshape(B, A),
            macro_cf_action_value=action_value.reshape(B, A),
            macro_cf_support_pos=support_pos.reshape(B, A),
            macro_cf_support_shuffled=support_shuffled.reshape(B, A),
            macro_cf_support_perturbed=support_perturbed.reshape(B, A, -1),
            macro_cf_perturbed_value=perturbed_value.reshape(B, A, -1),
            macro_cf_valid=batch["macro_alt_valid"],
            macro_cf_remaining=batch["macro_alt_remaining"],
            macro_cf_prefix_remaining=batch["macro_alt_prefix_remaining"],
            macro_cf_advantage=batch["macro_alt_advantage"],
            subgoal_action_cost=subgoal_action_cost,
            subgoal_action_cost_pred=subgoal_action_cost_pred,
            subgoal_action_positive=subgoal_action_positive,
        )

    def _geo_rank(self, batch: dict, out) -> None:
        """Geometric-advantage ranking: energies for executed + K alt
        actions at one anchor step; labels = LN-L1 distance of the TRUE
        next states (EMA-encoded text) to the EMA terminal goal. No
        symbolic annotations anywhere."""
        import torch.nn.functional as Fn

        B = out.s0.shape[0]
        device = out.s0.device
        t = batch["ga_t"].clamp(min=0)
        bidx = torch.arange(B, device=device)
        valid_b = batch["ga_t"] >= 0
        # model-side energies
        a_alt = self.encode_actions(batch["ga_alt_action_tokens"])  # [B,K,da]
        K = a_alt.shape[1]
        s_anchor = out.prev_states[bidx, t]  # state before the anchor step
        preds_alt = self.core.predictor(
            s_anchor.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1),
            a_alt.reshape(B * K, -1),
        )
        pe = out.preds[bidx, t]
        if self.core.value_detach:
            pe, preds_alt = pe.detach(), preds_alt.detach()
        e_exec = self.core.value_head(pe, out.s0)
        e_alt = self.core.value_head(
            preds_alt, out.s0.repeat_interleave(K, 0)
        ).reshape(B, K)
        # Geometric labels in EMA space.  For horizon 1 these are true
        # one-step next states.  Longer horizons either use random shooting or
        # an online geometry-greedy feasible-action policy.  Neither variant
        # consumes symbolic quality/order annotations.
        with torch.no_grad():
            last = out.step_mask.sum(1).clamp(min=1) - 1
            goal = out.step_states_tgt[bidx, last]
            ln = lambda x: Fn.layer_norm(x, x.shape[-1:])
            if batch.get("ga_greedy", False):
                d, candidate_valid = self._greedy_geo_labels(batch, goal)
                out.extras["ga_greedy_distance"] = d
            elif "ga_rollout_step_tokens" in batch:
                rt = batch["ga_rollout_step_tokens"]  # [B,C,R,Tr,L]
                rm = batch["ga_rollout_step_mask"]
                rv = batch["ga_rollout_valid"]
                _, C, R, Tr, L = rt.shape
                flat_mask = rm.reshape(B * C * R, Tr).clone()
                # Transformer attention cannot consume an entirely masked
                # padded rollout.  Give invalid rows one harmless pad chunk;
                # rv excludes their resulting representation below.
                empty = ~flat_mask.any(1)
                flat_mask[empty, 0] = True
                _, rollout_states = self.encode_states(
                    batch["prompt_tokens"].repeat_interleave(C * R, 0),
                    batch["prompt_mask"].repeat_interleave(C * R, 0),
                    rt.reshape(B * C * R, Tr, L), flat_mask,
                    teacher=True,
                )
                leaf_idx = flat_mask.sum(1).clamp(min=1) - 1
                leaf = rollout_states[
                    torch.arange(B * C * R, device=device), leaf_idx
                ].reshape(B, C, R, -1)
                d_rollout = (
                    ln(leaf) - ln(goal).view(B, 1, 1, -1)
                ).abs().mean(-1)
                d_rollout = d_rollout.masked_fill(~rv, float("inf"))
                d = d_rollout.amin(-1)
                candidate_valid = rv.any(-1) & valid_b.unsqueeze(1)
                out.extras["ga_rollout_distance"] = d_rollout
            else:
                st = batch["step_tokens"]  # [B, T, L]
                ga_st = batch["ga_alt_step_tokens"]  # [B, K, La]
                T, L = st.shape[1], max(st.shape[2], ga_st.shape[2])
                seq = torch.full(
                    (B, K, T, L), self.chunk_encoder.pad_id
                    if hasattr(self.chunk_encoder, "pad_id") else 0,
                    dtype=torch.long, device=device,
                )
                seq[:, :, :, : st.shape[2]] = st.unsqueeze(1).expand(
                    -1, K, -1, -1
                )[:, :, :, : st.shape[2]].clone()
                # place alt step at position t, blank positions after t
                ar = torch.arange(T, device=device).view(1, 1, T)
                after = ar > t.view(B, 1, 1)
                seq[after.unsqueeze(-1).expand_as(seq)] = seq.new_tensor(0)
                for b in range(B):
                    if batch["ga_t"][b] < 0:
                        continue
                    seq[b, :, t[b], :] = 0
                    seq[b, :, t[b], : ga_st.shape[2]] = ga_st[b]
                mask = (ar <= t.view(B, 1, 1)).expand(B, K, T)
                _, alt_states = self.encode_states(
                    batch["prompt_tokens"].repeat_interleave(K, 0),
                    batch["prompt_mask"].repeat_interleave(K, 0),
                    seq.reshape(B * K, T, L), mask.reshape(B * K, T),
                    teacher=True,
                )
                s_alt_true = alt_states.reshape(B, K, T, -1)[
                    bidx.unsqueeze(1),
                    torch.arange(K, device=device).unsqueeze(0),
                    t.unsqueeze(1),
                ]
                d_alt = (
                    ln(s_alt_true) - ln(goal).unsqueeze(1)
                ).abs().mean(-1)
                d_exec = (
                    ln(out.step_states_tgt[bidx, t]) - ln(goal)
                ).abs().mean(-1)
                d = torch.cat([d_exec.unsqueeze(1), d_alt], 1)
                candidate_valid = torch.cat(
                    [valid_b.unsqueeze(1),
                     batch["ga_valid"] & valid_b.unsqueeze(1)], 1
                )
        out.extras["ga_energy"] = torch.cat([e_exec.unsqueeze(1), e_alt], 1)
        out.extras["ga_label"] = d
        out.extras["ga_valid"] = candidate_valid

    @torch.no_grad()
    def _greedy_geo_labels(
        self, batch: dict, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approximate N-step action quality with a bounded geometric beam.

        For every initial candidate, execute the action in the environment and
        then repeatedly enumerate *all feasible* next actions. True rendered
        outcomes are EMA-encoded and the continuation retains the B children
        with minimum latent distance to the trace-terminal goal for each root
        action. B=1 exactly recovers the original greedy teacher. Feasibility
        and outcome text are the only environment signals: symbolic relevance,
        remaining-step counts, and action-quality labels are never consulted.
        """
        import torch.nn.functional as Fn

        device = goal.device
        vocab = batch["ga_vocab"]
        candidate_objects = batch.get("ga_candidate_objects")
        if candidate_objects is None:
            candidate_ids = batch["ga_candidate_ids"]
            B, C = candidate_ids.shape
            candidate_objects = [
                [int(candidate_ids[b, c].item()) for c in range(C)]
                for b in range(B)
            ]
            valid = candidate_ids >= 0
        else:
            B, C = len(candidate_objects), len(candidate_objects[0])
            valid = torch.tensor(
                [[action is not None for action in row]
                 for row in candidate_objects],
                dtype=torch.bool, device=device,
            )
        horizon = int(batch.get("ga_horizon", 1))
        beam_width = max(1, int(batch.get("ga_beam_width", 1)))
        paths: list[dict] = []

        for b in range(B):
            if int(batch["ga_t"][b].item()) < 0:
                continue
            problem = batch["ga_problems"][b]
            trace = batch["ga_traces"][b]
            if problem is None or trace is None:
                continue
            t = int(batch["ga_t"][b].item())
            env_kind = batch.get("ga_env_kinds", ["stylized"] * B)[b]
            if env_kind == "faithful":
                from textjepa.data.faithful import FaithfulEnv

                env = FaithfulEnv(problem)
            else:
                from textjepa.data.igsm.env import SymbolicEnv

                env = SymbolicEnv(problem)
            history: list[list[int]] = []
            for idx in trace[:t]:
                history.append(vocab.encode(env.step(idx)))
            for c in range(C):
                action = candidate_objects[b][c]
                if action is None:
                    continue
                child = env.clone()
                sequence = list(history)
                sequence.append(vocab.encode(child.step(action)))
                paths.append({"b": b, "c": c, "env": child, "seq": sequence})

        labels = goal.new_full((B, C), float("inf"))
        if not paths:
            return labels, valid & False

        leaf = list(self._encode_path_entries(batch, paths).unbind(0))
        ln = lambda x: Fn.layer_norm(x, x.shape[-1:])

        # The root action is depth one. At each later depth, retain a bounded
        # beam independently for every root candidate. Complexity is linear in
        # the configured beam width rather than exponential in the horizon.
        for _depth in range(1, horizon):
            options: list[dict] = []
            for p_i, path in enumerate(paths):
                env = path["env"]
                if env.solved:
                    options.append({
                        "b": path["b"], "c": path["c"], "env": env,
                        "seq": path["seq"], "leaf": leaf[p_i],
                    })
                    continue
                for action in env.feasible_actions():
                    child = env.clone()
                    sequence = list(path["seq"])
                    sequence.append(vocab.encode(child.step(action)))
                    options.append(
                        {"b": path["b"], "c": path["c"],
                         "env": child, "seq": sequence}
                    )
            if not options:
                break
            encode_indices = [
                i for i, option in enumerate(options) if "leaf" not in option
            ]
            if encode_indices:
                encoded = self._encode_path_entries(
                    batch, [options[i] for i in encode_indices]
                )
                for row, option_i in enumerate(encode_indices):
                    options[option_i]["leaf"] = encoded[row]
            option_leaf = torch.stack([option["leaf"] for option in options])
            option_b = torch.tensor(
                [o["b"] for o in options], device=device, dtype=torch.long
            )
            option_dist = (
                ln(option_leaf) - ln(goal.index_select(0, option_b))
            ).abs().mean(-1)
            by_root: dict[tuple[int, int], list[int]] = {}
            for option_i, option in enumerate(options):
                by_root.setdefault(
                    (int(option["b"]), int(option["c"])), []
                ).append(option_i)
            selected: list[int] = []
            for indices in by_root.values():
                indices.sort(key=lambda i: float(option_dist[i]))
                selected.extend(indices[:beam_width])
            paths = [
                {key: value for key, value in options[i].items()
                 if key != "leaf"}
                for i in selected
            ]
            leaf = [option_leaf[i] for i in selected]

        leaf_t = torch.stack(leaf)
        path_b = torch.tensor(
            [p["b"] for p in paths], device=device, dtype=torch.long
        )
        final_dist = (
            ln(leaf_t) - ln(goal.index_select(0, path_b))
        ).abs().mean(-1)
        for p_i, path in enumerate(paths):
            b, c = path["b"], path["c"]
            labels[b, c] = torch.minimum(labels[b, c], final_dist[p_i])
        return labels, valid & torch.isfinite(labels)

    def _encode_path_entries(
        self, batch: dict, entries: list[dict]
    ) -> torch.Tensor:
        """EMA-encode variable-length rendered histories in bounded chunks.

        Beam continuation can produce tens of thousands of path expansions
        from a full training batch. Feeding every expansion through one
        Transformer call can exceed CUDA attention-grid limits even when the
        tensors fit in memory. Chunking is label-preserving because target
        histories are encoded independently.
        """
        device = batch["prompt_tokens"].device
        pad = int(batch["ga_vocab"].pad_id)
        n = len(entries)
        T = max(len(e["seq"]) for e in entries)
        L = max(len(s) for e in entries for s in e["seq"])
        tokens = torch.full((n, T, L), pad, dtype=torch.long, device=device)
        mask = torch.zeros(n, T, dtype=torch.bool, device=device)
        for i, entry in enumerate(entries):
            for t, sentence in enumerate(entry["seq"]):
                tokens[i, t, : len(sentence)] = torch.tensor(
                    sentence, dtype=torch.long, device=device
                )
                mask[i, t] = True
        bidx = torch.tensor(
            [e["b"] for e in entries], dtype=torch.long, device=device
        )
        leaves = []
        encode_batch = 256
        for start in range(0, n, encode_batch):
            stop = min(start + encode_batch, n)
            _, states = self.encode_states(
                batch["prompt_tokens"].index_select(0, bidx[start:stop]),
                batch["prompt_mask"].index_select(0, bidx[start:stop]),
                tokens[start:stop], mask[start:stop], teacher=True,
            )
            last = mask[start:stop].sum(1).clamp(min=1) - 1
            leaves.append(states[
                torch.arange(stop - start, device=device), last
            ])
        return torch.cat(leaves, dim=0)
