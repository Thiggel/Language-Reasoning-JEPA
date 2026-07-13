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

from textjepa.models.action import ActionEncoder, VariationalAction
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
        n_ops: int = 4,
        macro_k: int = 3,
        d_macro: int = 8,
        value_detach: bool = True,
        dropout: float = 0.0,
        chunk_target: str = "frozen",  # "frozen" | "ema" anchor for chunk_pred
        freeze_encoders: bool = False,  # baseline: random frozen representation
        geo_proj: bool = False,  # geometry losses act on a learned projection
        # "ema" | "online" (sg online states) | "online_nosg" (no stopgrad —
        # the Delta-JEPA stability claim: LDAD alone prevents collapse)
        state_target: str = "ema",
        predictor_residual: bool = True,
        predictor_kind: str = "concat",
        variational_actions: bool = False,
    ):
        super().__init__()
        self.chunk_target = chunk_target
        self.freeze_encoders = freeze_encoders
        self.state_target = state_target
        self.chunk_encoder = TokenTransformer(
            vocab_size, pad_id, d_model, chunk_layers, chunk_heads,
            ff_mult, max_chunk_len, dropout,
        )
        self.state_model = DiscourseStateModel(
            d_model, state_layers, state_heads, ff_mult, max_chunks, dropout
        )
        self.action_encoder = ActionEncoder(d_model, d_action, fsq_levels=fsq_levels)
        self.var_action = (
            VariationalAction(d_model, d_action) if variational_actions else None
        )
        from textjepa.models.layers import mlp as _mlp

        self.act_decode = (
            _mlp([d_action, d_model], d_model) if variational_actions else None
        )
        self.core = LatentDynamicsCore(
            d_model, d_action, predictor_hidden_mult, predictor_layers,
            n_ops, macro_k, d_macro, value_detach, geo_proj,
            residual=predictor_residual,
            detach_targets=state_target != "online_nosg",
            predictor_kind=predictor_kind,
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
        return self.action_encoder(self.encode_chunks(action_tokens))

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
            actions = self.action_encoder(self.encode_chunks(batch["action_tokens"]))
        alt_actions = self._encode_alt(batch)
        out = self.core(
            s0, step_states, step_states_tgt, actions, action_emb_tgt,
            batch["step_mask"], step_emb_tgt=step_emb_tgt,
            alt_actions=alt_actions,
        )
        out.extras.update(var_extras)
        return out
