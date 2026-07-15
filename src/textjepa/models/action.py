"""Action bottlenecks: compressed step codes and macro-actions.

The action latent is deliberately tiny (HWM finds ~4-8 dims optimal): it
must carry the *intent* of a discourse move, not its content.
"""

from __future__ import annotations

import torch
from torch import nn

from textjepa.models.layers import encoder_stack, mlp


class FSQ(nn.Module):
    """Finite scalar quantization with a straight-through estimator."""

    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z)
        half = (self.levels - 1) / 2
        zq = torch.round(z * half) / half
        return z + (zq - z).detach()


class ActionEncoder(nn.Module):
    """Phrase embedding [.., D] -> small action latent [.., d_action]."""

    def __init__(
        self,
        d_model: int,
        d_action: int = 16,
        hidden_mult: int = 2,
        fsq_levels: list[int] | None = None,
    ):
        super().__init__()
        self.proj = mlp([d_model, d_model * hidden_mult // 2], d_action)
        self.quantizer = FSQ(fsq_levels) if fsq_levels else None

    def forward(self, phrase_emb: torch.Tensor) -> torch.Tensor:
        a = self.proj(phrase_emb)
        return self.quantizer(a) if self.quantizer else a


class TokenBottleneckActionEncoder(nn.Module):
    """Order-preserving action code from concatenated token embeddings.

    The standard :class:`ActionEncoder` first pools the entire intent phrase
    to one sentence vector.  This alternative projects every word embedding
    to a very small channel, concatenates the resulting position-specific
    vectors, and only then compresses the phrase to ``d_action``.  It is a
    controlled observed-action alternative to an inferred variational code:
    no outcome tokens or future state enter the encoder.
    """

    def __init__(
        self,
        d_model: int,
        d_action: int = 16,
        max_len: int = 48,
        token_dim: int = 8,
        fsq_levels: list[int] | None = None,
    ):
        super().__init__()
        self.max_len = max_len
        self.norm = nn.LayerNorm(d_model)
        self.token_proj = nn.Linear(d_model, token_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, token_dim))
        nn.init.normal_(self.pos, std=0.02)
        self.proj = mlp([max_len * token_dim, max(4 * d_action, 64)], d_action)
        self.quantizer = FSQ(fsq_levels) if fsq_levels else None

    def forward(
        self, token_emb: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """``token_emb`` [..., L, D], ``token_mask`` [..., L]."""
        if token_emb.shape[-2] > self.max_len:
            raise ValueError(
                f"action length {token_emb.shape[-2]} exceeds {self.max_len}"
            )
        shape = token_emb.shape[:-2]
        length = token_emb.shape[-2]
        h = self.token_proj(self.norm(token_emb)) + self.pos[:, :length]
        h = h * token_mask.unsqueeze(-1).to(h.dtype)
        if length < self.max_len:
            h = torch.nn.functional.pad(h, (0, 0, 0, self.max_len - length))
        action = self.proj(h.reshape(*shape, -1))
        return self.quantizer(action) if self.quantizer else action


class MacroActionEncoder(nn.Module):
    """CLS-transformer over K action latents -> macro-action [.., d_macro]."""

    def __init__(
        self,
        d_action: int,
        d_macro: int = 8,
        d_hidden: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.inp = nn.Linear(d_action, d_hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_hidden))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, 16, d_hidden))
        nn.init.normal_(self.pos, std=0.02)
        self.encoder = encoder_stack(d_hidden, n_layers, n_heads, 2, 0.0)
        self.out = nn.Linear(d_hidden, d_macro)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [N, K, d_action] -> [N, d_macro]."""
        h = self.inp(actions) + self.pos[:, 1 : actions.shape[1] + 1]
        h = torch.cat([self.cls.expand(h.shape[0], 1, -1) + self.pos[:, :1], h], 1)
        return self.out(self.encoder(h)[:, 0])


class ConcatMacroActionEncoder(nn.Module):
    """Order-preserving projected concatenation of a fixed action span."""

    def __init__(
        self,
        d_action: int,
        d_macro: int,
        span: int,
        action_width: int = 8,
    ):
        super().__init__()
        self.span = span
        self.norm = nn.LayerNorm(d_action)
        self.action_proj = nn.Linear(d_action, action_width)
        self.pos = nn.Parameter(torch.zeros(1, span, action_width))
        nn.init.normal_(self.pos, std=0.02)
        self.out = mlp(
            [span * action_width, max(4 * d_macro, 64)], d_macro
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-2] != self.span:
            raise ValueError(
                f"expected {self.span} lower-level actions, got "
                f"{actions.shape[-2]}"
            )
        h = self.action_proj(self.norm(actions)) + self.pos
        return self.out(h.flatten(-2))


class MacroActionModel(nn.Module):
    """Macro encoder plus a calibrated state-conditioned macro prior.

    The deterministic variant treats its encoded span as an observed target
    for ``p(m | s)``. The variational variant learns a Gaussian posterior over
    that same observed span and minimizes ``KL(q || p)``. Primitive actions
    and latent states remain deterministic.
    """

    def __init__(
        self,
        d_action: int,
        d_state: int,
        d_macro: int,
        span: int,
        kind: str = "transformer",
        variational: bool = False,
        concat_width: int = 8,
        hidden: int = 256,
    ):
        super().__init__()
        if kind == "transformer":
            self.encoder = MacroActionEncoder(d_action, d_macro)
        elif kind == "concat":
            self.encoder = ConcatMacroActionEncoder(
                d_action, d_macro, span, concat_width
            )
        else:
            raise ValueError(f"unknown macro encoder kind: {kind}")
        self.d_macro = d_macro
        self.variational = variational
        self.posterior_logvar = (
            nn.Linear(d_macro, d_macro) if variational else None
        )
        if self.posterior_logvar is not None:
            nn.init.zeros_(self.posterior_logvar.weight)
            nn.init.constant_(self.posterior_logvar.bias, -2.0)
        self.prior = mlp([d_state, hidden], 2 * d_macro)

    @staticmethod
    def _split(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = raw.chunk(2, -1)
        return mu, logvar.clamp(-6.0, 2.0)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Posterior mean, preserving the historical deterministic API."""
        return self.encoder(actions)

    def prior_params(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.prior(state))

    def sample_prior(self, state: torch.Tensor, n: int = 1) -> torch.Tensor:
        mu, logvar = self.prior_params(state)
        if n == 1:
            return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        eps = torch.randn(
            *mu.shape[:-1], n, mu.shape[-1], device=mu.device
        )
        return mu.unsqueeze(-2) + eps * (0.5 * logvar).exp().unsqueeze(-2)

    def training_code(
        self, actions: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_mu = self.encoder(actions)
        p_mu, p_logvar = self.prior_params(state)
        if self.variational:
            q_logvar = self.posterior_logvar(q_mu).clamp(-6.0, 2.0)
            code = q_mu + torch.randn_like(q_mu) * (0.5 * q_logvar).exp()
            prior_loss = 0.5 * (
                p_logvar - q_logvar
                + (q_logvar.exp() + (q_mu - p_mu).square())
                / p_logvar.exp()
                - 1.0
            ).sum(-1)
        else:
            q_logvar = torch.full_like(q_mu, -8.0)
            code = q_mu
            # The density fit must not collapse the action encoder. Hierarchy
            # prediction remains responsible for the macro representation.
            prior_loss = 0.5 * (
                p_logvar
                + (q_mu.detach() - p_mu).square() * (-p_logvar).exp()
            ).sum(-1)
        return code, {
            "macro_q_mu": q_mu,
            "macro_q_logvar": q_logvar,
            "macro_p_mu": p_mu,
            "macro_p_logvar": p_logvar,
            "macro_prior_loss": prior_loss,
        }


class VariationalAction(nn.Module):
    """Unobserved latent actions: posterior q(a | s_prev, s_next) infers
    the action code from the observed transition; prior p(a | s_prev)
    proposes codes at plan time. Reparametrized Gaussian, 16-d."""

    def __init__(self, d_state: int, d_action: int, hidden: int = 256):
        super().__init__()
        self.post = mlp([2 * d_state, hidden], 2 * d_action)
        self.prior = mlp([d_state, hidden], 2 * d_action)

    @staticmethod
    def _split(x):
        mu, logvar = x.chunk(2, dim=-1)
        return mu, logvar.clamp(-6, 2)

    def sample_posterior(self, s_prev, s_next):
        mu, logvar = self._split(self.post(torch.cat([s_prev, s_next], -1)))
        a = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return a, (mu, logvar)

    def prior_params(self, s_prev):
        return self._split(self.prior(s_prev))

    def sample_prior(self, s_prev, k: int = 1):
        mu, logvar = self.prior_params(s_prev)
        std = (0.5 * logvar).exp()
        if k == 1:
            return mu + torch.randn_like(mu) * std
        return mu.unsqueeze(-2) + torch.randn(
            *mu.shape[:-1], k, mu.shape[-1], device=mu.device
        ) * std.unsqueeze(-2)

    @staticmethod
    def kl(q_params, p_params):
        qm, ql = q_params
        pm, pl = p_params
        return 0.5 * (
            pl - ql + (ql.exp() + (qm - pm) ** 2) / pl.exp() - 1
        ).sum(-1)


class MixtureVariationalAction(VariationalAction):
    """Gaussian posterior with a context-conditioned Gaussian-mixture prior.

    A single Gaussian prior must blur together distinct feasible continuations
    from the same state.  This controlled alternative leaves the transition-
    informed posterior unchanged and makes only the plan-time prior multimodal.
    The training objective uses the standard log-sum-exp upper bound on
    ``KL(q || sum_k pi_k p_k)`` built from analytic Gaussian component KLs.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int,
        hidden: int = 256,
        n_components: int = 4,
    ):
        if n_components < 2:
            raise ValueError("mixture prior requires at least two components")
        super().__init__(d_state, d_action, hidden)
        self.d_action = d_action
        self.n_components = n_components
        self.prior = mlp(
            [d_state, hidden], n_components * (1 + 2 * d_action)
        )

    def prior_components(self, s_prev):
        raw = self.prior(s_prev).reshape(
            *s_prev.shape[:-1], self.n_components, 1 + 2 * self.d_action
        )
        logits = raw[..., 0]
        mu = raw[..., 1 : 1 + self.d_action]
        logvar = raw[..., 1 + self.d_action :].clamp(-6, 2)
        return logits, mu, logvar

    def prior_params(self, s_prev):
        logits, mu, logvar = self.prior_components(s_prev)
        weight = logits.softmax(-1).unsqueeze(-1)
        mean = (weight * mu).sum(-2)
        second = (weight * (logvar.exp() + mu.square())).sum(-2)
        variance = (second - mean.square()).clamp_min(1e-6)
        # The first two entries preserve the probe/report interface: they are
        # the exact first two moments of the mixture, not one selected mode.
        return mean, variance.log(), logits, mu, logvar

    def sample_prior(self, s_prev, k: int = 1):
        _, _, logits, mu, logvar = self.prior_params(s_prev)
        prefix = logits.shape[:-1]
        index = torch.multinomial(
            logits.softmax(-1).reshape(-1, self.n_components),
            num_samples=k,
            replacement=True,
        ).reshape(*prefix, k)
        gather = index.unsqueeze(-1).expand(*index.shape, self.d_action)
        selected_mu = torch.gather(mu, -2, gather)
        selected_lv = torch.gather(logvar, -2, gather)
        sample = selected_mu + torch.randn_like(selected_mu) * (
            0.5 * selected_lv
        ).exp()
        return sample.squeeze(-2) if k == 1 else sample

    @staticmethod
    def kl(q_params, p_params):
        if len(p_params) == 2:
            return VariationalAction.kl(q_params, p_params)
        qm, ql = q_params
        _, _, logits, pm, pl = p_params
        component_kl = 0.5 * (
            pl
            - ql.unsqueeze(-2)
            + (
                ql.exp().unsqueeze(-2)
                + (qm.unsqueeze(-2) - pm).square()
            )
            / pl.exp()
            - 1
        ).sum(-1)
        return -torch.logsumexp(logits.log_softmax(-1) - component_kl, -1)
