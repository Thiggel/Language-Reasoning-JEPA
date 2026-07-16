"""Variational JEPA over a single stream of ordinary text sentences.

Prompt and solution sentences are packed into one causal sequence.  There is
no intent-phrase input, symbolic action label, or feasible-action interface.
Each observed adjacent transition infers an unobserved action ``u_t`` and a
distribution over the next latent state.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.action import MixtureVariationalAction, VariationalAction
from textjepa.models.ema import EMATeacher
from textjepa.models.layers import TokenTransformer, mlp
from textjepa.models.outputs import JEPAOutputs
from textjepa.models.predictor import ProbabilisticActionConditionedPredictor
from textjepa.models.state_model import CausalSentenceStateModel


def pack_sentence_stream(batch: dict, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack valid prompt and solution chunks without retaining a boundary."""
    prompt, pm = batch["prompt_tokens"], batch["prompt_mask"]
    steps, sm = batch["step_tokens"], batch["step_mask"]
    lengths = pm.sum(1) + sm.sum(1)
    B, S = prompt.shape[0], int(lengths.max().item())
    L = max(prompt.shape[-1], steps.shape[-1])
    stream = prompt.new_full((B, S, L), pad_id)
    mask = torch.zeros(B, S, dtype=torch.bool, device=prompt.device)
    for b in range(B):
        p = prompt[b, pm[b]]
        t = steps[b, sm[b]]
        n_p, n_t = p.shape[0], t.shape[0]
        stream[b, :n_p, :p.shape[-1]] = p
        stream[b, n_p:n_p + n_t, :t.shape[-1]] = t
        mask[b, :n_p + n_t] = True
    return stream, mask


class SentenceStreamVJEPA(nn.Module):
    """Distributional sentence dynamics with fully latent actions.

    ``q(u_t | s_t, s_{t+1})`` infers a transition code, while
    ``p(u_t | s_t)`` is its context-conditioned prior.  Independently,
    ``q(z_{t+1}|x_{t+1})`` is a diagonal-Gaussian EMA target distribution and
    ``p(z_{t+1}|s_t,u_t)`` is a learned diagonal-Gaussian predictor.
    """

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
        max_chunk_len: int = 96,
        max_chunks: int = 96,
        d_action: int = 16,
        action_prior_components: int = 1,
        predictor_hidden_mult: int = 4,
        predictor_layers: int = 2,
        dropout: float = 0.0,
        target_logvar_init: float = -4.0,
        use_ema_target: bool = True,
        target_mode: str | None = None,
        latent_ldad: bool = False,
        counterfactual_set: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        # Backward compatibility: the first six pilots used a boolean that
        # conflated EMA with stop-gradient.  The explicit mode completes the
        # factorial with an online stop-gradient target.
        self.target_mode = target_mode or (
            "ema" if use_ema_target else "online_grad"
        )
        if self.target_mode not in {"ema", "online_sg", "online_grad"}:
            raise ValueError(f"unknown target_mode: {self.target_mode}")
        self.use_ema_target = self.target_mode == "ema"
        self.counterfactual_set = counterfactual_set
        self.chunk_encoder = TokenTransformer(
            vocab_size, pad_id, d_model, chunk_layers, chunk_heads,
            ff_mult, max_chunk_len, dropout,
        )
        self.state_model = CausalSentenceStateModel(
            d_model, state_layers, state_heads, ff_mult, max_chunks, dropout,
        )
        self.chunk_teacher = EMATeacher(self.chunk_encoder)
        self.state_teacher = EMATeacher(self.state_model)
        self.var_action = (
            MixtureVariationalAction(
                d_model, d_action, n_components=action_prior_components
            )
            if action_prior_components > 1
            else VariationalAction(d_model, d_action)
        )
        self.transition = ProbabilisticActionConditionedPredictor(
            d_model, d_action, predictor_hidden_mult, predictor_layers,
        )
        # V-JEPA's target mean follows EMA; its uncertainty remains trainable.
        self.target_logvar = mlp([d_model, d_model], d_model)
        nn.init.zeros_(self.target_logvar[-1].weight)
        nn.init.constant_(self.target_logvar[-1].bias, target_logvar_init)
        self.latent_action_decoder = (
            mlp([d_model, d_model], d_action) if latent_ldad else None
        )

    def _encode(self, tokens: torch.Tensor, mask: torch.Tensor, teacher=False):
        B, S, L = tokens.shape
        chunk = self.chunk_teacher if teacher else self.chunk_encoder
        state = self.state_teacher if teacher else self.state_model
        emb = chunk(tokens.reshape(B * S, L)).reshape(B, S, -1)
        return state(emb, mask)

    @torch.no_grad()
    def update_teachers(self, momentum: float) -> None:
        self.chunk_teacher.update(self.chunk_encoder, momentum)
        self.state_teacher.update(self.state_model, momentum)

    def forward(self, batch: dict) -> JEPAOutputs:
        tokens, sentence_mask = pack_sentence_stream(batch, self.pad_id)
        states = self._encode(tokens, sentence_mask)
        if self.target_mode == "ema":
            with torch.no_grad():
                target_mu_all = self._encode(tokens, sentence_mask, teacher=True)
        elif self.target_mode == "online_sg":
            target_mu_all = states.detach()
        else:  # online_grad
            # Symmetric, end-to-end target: the same encoder, without EMA or
            # stop-gradient.
            target_mu_all = states

        prev = states[:, :-1]
        next_online = states[:, 1:]
        target_mu = target_mu_all[:, 1:]
        if self.target_mode in {"ema", "online_sg"}:
            target_mu = target_mu.detach()
        trans_mask = sentence_mask[:, :-1] & sentence_mask[:, 1:]

        # Target encoder inference distribution q(z_next | observed sentence).
        target_lv = self.target_logvar(target_mu).clamp(-8.0, 3.0)
        target_sample = target_mu + torch.randn_like(target_mu) * (
            0.5 * target_lv
        ).exp()

        # Fully unobserved transition variable q(u|s,s') and its plan-time prior.
        actions, q_u = self.var_action.sample_posterior(prev, target_sample.detach())
        p_u = self.var_action.prior_params(prev.detach())
        pred_mu, pred_lv = self.transition(prev, actions)

        cur, rollout = states[:, 0], []
        for t in range(actions.shape[1]):
            cur, _ = self.transition(cur, actions[:, t])
            rollout.append(cur)
        rollout_mu = torch.stack(rollout, 1)

        target_kl = 0.5 * (
            target_mu.pow(2) + target_lv.exp() - 1.0 - target_lv
        ).mean(-1)
        extras = {
            "sentence_stream": True,
            "pred_logvar": pred_lv,
            "target_logvar": target_lv,
            "target_sample": target_sample,
            "target_kl": target_kl,
            "action_kl": self.var_action.kl(q_u, p_u),
            "action_q_mu": q_u[0],
            "action_q_logvar": q_u[1],
            "action_p_mu": p_u[0],
            "action_p_logvar": p_u[1],
            "sentence_mask": sentence_mask,
            "uses_ema_target": self.use_ema_target,
            "target_mode": self.target_mode,
        }
        if self.counterfactual_set:
            extras.update(self._counterfactual_set_terms(batch, states))
        if self.latent_action_decoder is not None:
            # Closest identifiable LDAD analogue when actions are unobserved:
            # reconstruct a stop-gradient posterior action from the *online*
            # latent displacement.  Unlike Delta-JEPA's observed raw-action
            # target this remains vulnerable to posterior collusion, which is
            # exactly why the VICReg/SIGReg factorial is reported explicitly.
            extras["latent_ldad_pred"] = self.latent_action_decoder(
                next_online - prev
            )
            extras["latent_ldad_tgt"] = q_u[0].detach()

        B, T, D = pred_mu.shape
        zeros_d = pred_mu.new_zeros(B, T, D)
        return JEPAOutputs(
            s0=states[:, 0],
            step_states=next_online,
            prev_states=prev,
            step_states_tgt=target_mu,
            actions=actions,
            action_emb_tgt=zeros_d,
            preds=pred_mu,
            rollout=rollout_mu,
            op_logits=pred_mu.new_zeros(B, T, 4),
            emb_pred=zeros_d,
            value_pred=pred_mu.new_zeros(B, T + 1),
            step_mask=trans_mask,
            extras=extras,
        )

    def _counterfactual_set_terms(
        self, batch: dict, states: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """ELBO terms for every rendered same-state alternative outcome.

        This auxiliary receives no intent phrase, action identifier, quality
        score, or relevance label.  For each demonstrated solution state it
        sees only the set of alternative next sentences supplied by
        environment interaction.  The transition-informed posterior must
        explain each outcome, while the plan-time prior is trained against
        every resulting posterior mode rather than only the demonstrated one.
        """
        if "alt_step_tokens" not in batch or "alt_remaining" not in batch:
            zero = states.sum() * 0.0
            return {
                "counterfactual_variational_nll": zero.reshape(1),
                "counterfactual_action_kl": zero.reshape(1),
                "counterfactual_set_count": states.new_zeros(1),
            }

        valid = (batch["alt_remaining"] >= 0) & batch["step_mask"].unsqueeze(-1)
        indices = valid.nonzero(as_tuple=False)
        if indices.numel() == 0:
            zero = states.sum() * 0.0
            return {
                "counterfactual_variational_nll": zero.reshape(1),
                "counterfactual_action_kl": zero.reshape(1),
                "counterfactual_set_count": states.new_zeros(1),
            }

        device = states.device
        n = indices.shape[0]
        prompt_width = batch["prompt_tokens"].shape[-1]
        step_width = batch["step_tokens"].shape[-1]
        alt_width = batch["alt_step_tokens"].shape[-1]
        token_width = max(prompt_width, step_width, alt_width)
        prompt_lengths = batch["prompt_mask"].sum(1).long()
        sequence_lengths = prompt_lengths.index_select(0, indices[:, 0]) + indices[:, 1] + 1
        max_sentences = int(sequence_lengths.max().item())
        streams = batch["prompt_tokens"].new_full(
            (n, max_sentences, token_width), self.pad_id
        )
        masks = torch.zeros(
            n, max_sentences, dtype=torch.bool, device=device
        )

        for row, (b_tensor, t_tensor, k_tensor) in enumerate(indices):
            b, t, k = int(b_tensor), int(t_tensor), int(k_tensor)
            n_prompt = int(prompt_lengths[b])
            streams[row, :n_prompt, :prompt_width] = batch["prompt_tokens"][
                b, :n_prompt
            ]
            if t:
                streams[
                    row, n_prompt : n_prompt + t, :step_width
                ] = batch["step_tokens"][b, :t]
            streams[row, n_prompt + t, :alt_width] = batch[
                "alt_step_tokens"
            ][b, t, k]
            masks[row, : n_prompt + t + 1] = True

        teacher = self.target_mode == "ema"
        if self.target_mode in {"ema", "online_sg"}:
            with torch.no_grad():
                alternative_states = self._encode_stream_batches(
                    streams, masks, teacher=teacher
                )
        else:
            alternative_states = self._encode_stream_batches(
                streams, masks, teacher=False
            )
        rows = torch.arange(n, device=device)
        alternative_target = alternative_states[rows, sequence_lengths - 1]
        if self.target_mode in {"ema", "online_sg"}:
            alternative_target = alternative_target.detach()

        b = indices[:, 0]
        t = indices[:, 1]
        current_index = prompt_lengths.index_select(0, b) - 1 + t
        current = states[b, current_index]
        target_logvar = self.target_logvar(alternative_target).clamp(-8.0, 3.0)
        target_sample = alternative_target + torch.randn_like(alternative_target) * (
            0.5 * target_logvar
        ).exp()
        actions, posterior = self.var_action.sample_posterior(
            current, target_sample.detach()
        )
        prior = self.var_action.prior_params(current.detach())
        pred_mu, pred_logvar = self.transition(current, actions)
        nll = 0.5 * (
            pred_logvar
            + (target_sample - pred_mu).pow(2) * torch.exp(-pred_logvar)
        ).mean(-1)
        return {
            "counterfactual_variational_nll": nll,
            "counterfactual_action_kl": self.var_action.kl(posterior, prior),
            "counterfactual_set_count": states.new_tensor([float(n)]),
        }

    def _encode_stream_batches(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        teacher: bool,
        max_streams: int = 256,
    ) -> torch.Tensor:
        """Encode independent histories without exceeding CUDA grid limits.

        Exhaustive outcome sets can create thousands of histories from one
        training batch.  The chunk encoder internally flattens histories and
        sentences, so one monolithic attention call can exceed the CUDA SDPA
        launch grid even when memory is sufficient.  Histories do not attend
        across examples, making this split exactly equivalent.
        """
        return torch.cat(
            [
                self._encode(
                    tokens[start : start + max_streams],
                    mask[start : start + max_streams],
                    teacher=teacher,
                )
                for start in range(0, tokens.shape[0], max_streams)
            ],
            dim=0,
        )
