"""Probabilistic discourse JEPA with controlled action observability.

Unlike :mod:`sentence_vjepa`, this model keeps the iGSM prompt/solution and
intent/outcome interfaces fixed.  The only factorial variable is how the
transition is conditioned: an inferred latent action, a pooled observed
intent phrase, or an order-preserving token-bottleneck intent.  This makes the
effect of action observability identifiable while retaining a variational
next-state distribution in every cell.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import (
    ActionEncoder,
    TokenBottleneckActionEncoder,
    VariationalAction,
)
from textjepa.models.delta_decoder import ObservedActionDecoder
from textjepa.models.ema import EMATeacher
from textjepa.models.layers import TokenTransformer, mlp
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.predictor import ProbabilisticActionConditionedPredictor
from textjepa.models.state_model import DiscourseStateModel


class DiscourseVJEPA(nn.Module):
    """Diagonal-Gaussian next-state dynamics over intent/action transitions."""

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
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        dropout: float = 0.0,
        target_logvar_init: float = -4.0,
        target_mode: str = "ema",
        action_mode: str = "latent",  # latent | pooled | token_bottleneck
        action_token_dim: int = 8,
        predictor_residual: bool = True,
        observed_action_ldad: bool = False,
        latent_ldad: bool = False,
        ldad_decoder_layers: int = 2,
    ):
        super().__init__()
        if target_mode not in {"ema", "online_sg", "online_grad"}:
            raise ValueError(f"unknown target_mode: {target_mode}")
        if action_mode not in {"latent", "pooled", "token_bottleneck"}:
            raise ValueError(f"unknown action_mode: {action_mode}")
        if action_mode == "latent" and observed_action_ldad:
            raise ValueError("observed-action LDAD requires an observed action mode")
        self.pad_id = pad_id
        self.target_mode = target_mode
        self.action_mode = action_mode
        self.chunk_encoder = TokenTransformer(
            vocab_size, pad_id, d_model, chunk_layers, chunk_heads,
            ff_mult, max_chunk_len, dropout,
        )
        self.state_model = DiscourseStateModel(
            d_model, state_layers, state_heads, ff_mult, max_chunks, dropout,
        )
        self.chunk_teacher = EMATeacher(self.chunk_encoder)
        self.state_teacher = EMATeacher(self.state_model)
        if action_mode == "latent":
            self.action_encoder = None
            self.var_action = VariationalAction(d_model, d_action)
        elif action_mode == "pooled":
            self.action_encoder = ActionEncoder(d_model, d_action)
            self.var_action = None
        else:
            self.action_encoder = TokenBottleneckActionEncoder(
                d_model, d_action, max_chunk_len, action_token_dim
            )
            self.var_action = None
        self.transition = ProbabilisticActionConditionedPredictor(
            d_model, d_action, predictor_hidden_mult, predictor_layers,
            residual=predictor_residual,
        )
        self.target_logvar = mlp([d_model, d_model], d_model)
        nn.init.zeros_(self.target_logvar[-1].weight)
        nn.init.constant_(self.target_logvar[-1].bias, target_logvar_init)
        self.observed_action_decoder = (
            ObservedActionDecoder(
                d_model, vocab_size, max_chunk_len,
                n_layers=ldad_decoder_layers, n_heads=chunk_heads,
            )
            if observed_action_ldad else None
        )
        self.latent_action_decoder = (
            mlp([d_model, d_model], d_action) if latent_ldad else None
        )

    def encode_chunks(self, tokens: torch.Tensor, teacher: bool = False):
        B, C, L = tokens.shape
        encoder = self.chunk_teacher if teacher else self.chunk_encoder
        return encoder(tokens.reshape(B * C, L)).reshape(B, C, -1)

    def encode_states(self, batch: dict, teacher: bool = False):
        prompt = self.encode_chunks(batch["prompt_tokens"], teacher)
        steps = self.encode_chunks(batch["step_tokens"], teacher)
        state = self.state_teacher if teacher else self.state_model
        return state(prompt, batch["prompt_mask"], steps, batch["step_mask"])

    def encode_observed_actions(self, tokens: torch.Tensor):
        if self.action_mode == "pooled":
            return self.action_encoder(self.encode_chunks(tokens))
        token_emb = self.chunk_encoder.tok(tokens)
        return self.action_encoder(token_emb, tokens.ne(self.pad_id))

    @torch.no_grad()
    def update_teachers(self, momentum: float) -> None:
        self.chunk_teacher.update(self.chunk_encoder, momentum)
        self.state_teacher.update(self.state_model, momentum)

    def forward(self, batch: dict) -> JEPAOutputs:
        s0, states = self.encode_states(batch)
        if self.target_mode == "ema":
            with torch.no_grad():
                _, target_all = self.encode_states(batch, teacher=True)
        elif self.target_mode == "online_sg":
            target_all = states.detach()
        else:
            target_all = states
        target_mu = target_all.detach() if self.target_mode != "online_grad" else target_all
        prev = torch.cat([s0.unsqueeze(1), states[:, :-1]], dim=1)
        target_lv = self.target_logvar(target_mu).clamp(-8.0, 3.0)
        target_sample = target_mu + torch.randn_like(target_mu) * (
            0.5 * target_lv
        ).exp()

        if self.action_mode == "latent":
            actions, q_u = self.var_action.sample_posterior(
                prev, target_sample.detach()
            )
            p_u = self.var_action.prior_params(prev.detach())
            action_kl = self.var_action.kl(q_u, p_u)
            q_mu, p_mu = q_u[0], p_u[0]
        else:
            actions = self.encode_observed_actions(batch["action_tokens"])
            q_mu = p_mu = actions
            action_kl = actions.new_zeros(actions.shape[:-1])

        pred_mu, pred_lv = self.transition(prev, actions)
        cur, rollout = s0, []
        for t in range(actions.shape[1]):
            cur, _ = self.transition(cur, actions[:, t])
            rollout.append(cur)
        rollout_mu = torch.stack(rollout, dim=1)
        target_kl = 0.5 * (
            target_mu.pow(2) + target_lv.exp() - 1.0 - target_lv
        ).mean(-1)
        extras = {
            "sentence_stream": True,
            "discourse_variational": True,
            "pred_logvar": pred_lv,
            "target_logvar": target_lv,
            "target_sample": target_sample,
            "target_kl": target_kl,
            "action_kl": action_kl,
            "action_q_mu": q_mu,
            "action_p_mu": p_mu,
            "action_mode": self.action_mode,
            "target_mode": self.target_mode,
        }
        if self.action_mode == "latent":
            extras["action_q_logvar"] = q_u[1]
            extras["action_p_logvar"] = p_u[1]
        displacement = states - prev
        if self.observed_action_decoder is not None:
            extras["observed_action_logits"] = self.observed_action_decoder(
                displacement
            )
        if self.latent_action_decoder is not None:
            extras["latent_ldad_pred"] = self.latent_action_decoder(displacement)
            extras["latent_ldad_tgt"] = q_mu.detach()

        B, T, D = states.shape
        zeros = states.new_zeros(B, T, D)
        return JEPAOutputs(
            s0=s0,
            step_states=states,
            prev_states=prev,
            step_states_tgt=target_mu,
            actions=actions,
            action_emb_tgt=zeros,
            preds=pred_mu,
            rollout=rollout_mu,
            op_logits=states.new_zeros(B, T, 4),
            emb_pred=zeros,
            value_pred=states.new_zeros(B, T + 1),
            step_mask=batch["step_mask"],
            extras=extras,
        )
