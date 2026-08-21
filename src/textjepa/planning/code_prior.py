"""A LEARNED, STATE-CONDITIONED PRIOR OVER DISCRETE ACTION CODES.

Target architecture (project owner): planning happens entirely in latent
space -- a prior proposes action codes given the current state, the predictor
rolls them forward, the energy head scores them, and text is produced only at
execution time by a DETACHED decoder
(:mod:`textjepa.planning.action_decoder`).  The model never sees a menu, an
action list, or a feasibility signal.

This module is the prior.  It is discrete (a codebook) rather than a density,
because a codebook sidesteps by construction the diversity collapse that kills
the generative proposer: 16 samples at temperature 1.3 from the token head
collapse to ~1.6 distinct phrases, whereas K distinct codes are distinct by
definition.  Whether that distinctness SURVIVES decoding is an empirical
question this module is built to answer, not to assume.

Self-supervision, and why this is not a symbolic head
-----------------------------------------------------
Two things are learned, both from the actions that ACTUALLY OCCURRED in the
demonstrated traces:

1. the codebook itself, by vector quantization of the frozen checkpoint's own
   action vectors (an unsupervised clustering of the representation);
2. ``p(code | state)``, by cross-entropy against the code index the observed
   action quantizes to.

The target index is a LEARNED latent, not a label from the environment.  No
symbolic state, no steps-to-go, no feasibility supervision enters the
objective -- cf. the no-symbolic-heads rule in CLAUDE.md.  (Symbolic state is
used only to walk the demonstrated trace when building the data, exactly as
``scripts/train_flow_prior.py`` already does, and never as a model input.)

The backbone is frozen throughout; nothing here adds a config key.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

Tensor = torch.Tensor


@dataclass
class CodePriorConfig:
    dim: int
    n_codes: int = 256
    hidden: int = 1024
    n_layers: int = 3
    ema_decay: float = 0.99
    # Weight of the residual head; see ``dequantize``.
    residual: bool = True
    # Context-conditioned code ranking: cross-attend from the pooled state
    # (one query) over the frozen encoder's context hidden states, exactly
    # the pattern that fixed the detached decoder (entity names live in the
    # prompt, not in the pooled state).  Default off = the original MLP
    # prior; old checkpoints load unchanged.
    use_context: bool = False
    ctx_d_model: int = 384
    ctx_layers: int = 2
    ctx_heads: int = 6


class CodePrior(nn.Module):
    """Codebook over action vectors + p(code | state).

    Whitening statistics are buffers of the module (as in
    ``flow_prior.ConditionalFlowPrior``) so a saved prior is self-contained.
    """

    def __init__(self, cfg: CodePriorConfig):
        super().__init__()
        self.cfg = cfg
        d, K = cfg.dim, cfg.n_codes
        self.register_buffer("a_mean", torch.zeros(d))
        self.register_buffer("a_std", torch.ones(d))
        self.register_buffer("s_mean", torch.zeros(d))
        self.register_buffer("s_std", torch.ones(d))
        # EMA codebook (VQ-VAE v2 style): more stable than gradient updates
        # when the encoder is frozen and the "encoder output" is fixed data.
        self.register_buffer("codebook", torch.randn(K, d) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("code_sum", torch.zeros(K, d))
        layers: list[nn.Module] = [nn.Linear(d, cfg.hidden), nn.SiLU()]
        for _ in range(cfg.n_layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.logit_head = nn.Linear(cfg.hidden, K)
        # Residual head: the centroid alone is a coarse role code.  Given the
        # state AND the chosen code, predict the offset back to a specific
        # action vector, so distinct codes at the same state stay distinct
        # after dequantization.
        self.code_emb = nn.Embedding(K, cfg.hidden) if cfg.residual else None
        self.residual_head = (
            nn.Sequential(nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
                          nn.Linear(cfg.hidden, d))
            if cfg.residual else None
        )
        if self.residual_head is not None:
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        # Context branch (see CodePriorConfig.use_context).  Mirrors the
        # detached decoder: LayerNorm+Linear projections, a small
        # TransformerDecoder whose single query is the state, memory is the
        # projected context tokens.
        if cfg.use_context:
            dm = cfg.ctx_d_model
            self.s_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dm))
            self.ctx_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, dm))
            layer = nn.TransformerDecoderLayer(
                dm, cfg.ctx_heads, dm * 4, dropout=0.0, activation="gelu",
                batch_first=True, norm_first=True)
            self.ctx_dec = nn.TransformerDecoder(
                layer, cfg.ctx_layers, norm=nn.LayerNorm(dm))
            self.ctx_logit_head = nn.Linear(dm, K)
        else:
            self.s_proj = self.ctx_proj = None
            self.ctx_dec = self.ctx_logit_head = None

    # ------------------------------------------------------------ scaling
    @torch.no_grad()
    def fit_scaling(self, actions: Tensor, states: Tensor) -> None:
        self.a_mean.copy_(actions.mean(0))
        self.a_std.copy_(actions.std(0).clamp_min(1e-5))
        self.s_mean.copy_(states.mean(0))
        self.s_std.copy_(states.std(0).clamp_min(1e-5))

    def _wa(self, a: Tensor) -> Tensor:
        return (a - self.a_mean) / self.a_std

    def _ws(self, s: Tensor) -> Tensor:
        return (s - self.s_mean) / self.s_std

    @torch.no_grad()
    def init_codebook(self, actions: Tensor, iters: int = 25,
                      generator: torch.Generator | None = None) -> None:
        """k-means++ style init on whitened actions (Lloyd, then EMA takes over)."""
        w = self._wa(actions)
        n, K = w.shape[0], self.cfg.n_codes
        idx = torch.randperm(n, generator=generator)[:K]
        c = w[idx].clone()
        if c.shape[0] < K:  # fewer points than codes: pad with noise
            c = torch.cat([c, torch.randn(K - c.shape[0], w.shape[1]) * 0.1])
        for _ in range(iters):
            assign = torch.cdist(w, c).argmin(1)
            for k in range(K):
                m = assign == k
                if bool(m.any()):
                    c[k] = w[m].mean(0)
        self.codebook.copy_(c)
        assign = torch.cdist(w, c).argmin(1)
        self.cluster_size.copy_(
            torch.bincount(assign, minlength=K).float())
        self.code_sum.copy_(
            torch.zeros_like(self.code_sum).index_add_(0, assign, w))

    # ------------------------------------------------------- quantization
    @torch.no_grad()
    def quantize(self, actions: Tensor) -> Tensor:
        """Whitened action vectors -> nearest code index [B]."""
        return torch.cdist(self._wa(actions), self.codebook).argmin(1)

    @torch.no_grad()
    def ema_update(self, actions: Tensor) -> None:
        w = self._wa(actions)
        idx = torch.cdist(w, self.codebook).argmin(1)
        K = self.cfg.n_codes
        counts = torch.bincount(idx, minlength=K).float()
        sums = torch.zeros_like(self.code_sum).index_add_(0, idx, w)
        d = self.cfg.ema_decay
        self.cluster_size.mul_(d).add_(counts, alpha=1 - d)
        self.code_sum.mul_(d).add_(sums, alpha=1 - d)
        live = self.cluster_size > 1e-3
        self.codebook[live] = (
            self.code_sum[live] / self.cluster_size[live].unsqueeze(1))

    # -------------------------------------------------------------- prior
    def logits(self, states: Tensor, ctx: Tensor | None = None,
               ctx_mask: Tensor | None = None) -> Tensor:
        """p(code | state[, context]) logits [B, K].

        ``ctx`` [B, L, d] are frozen encoder hidden states over the prompt
        history, ``ctx_mask`` [B, L] is True at PAD positions.  A
        non-context prior silently ignores both, so call sites can always
        pass them.
        """
        if self.cfg.use_context:
            if ctx is None:
                raise ValueError("this CodePrior is context-conditioned; "
                                 "pass ctx (and ctx_mask)")
            if ctx_mask is None:
                ctx_mask = torch.zeros(ctx.shape[0], ctx.shape[1],
                                       dtype=torch.bool, device=ctx.device)
            q = self.s_proj(states).unsqueeze(1)
            mem = self.ctx_proj(ctx)
            out = self.ctx_dec(q, mem, memory_key_padding_mask=ctx_mask)
            return self.ctx_logit_head(out[:, 0])
        return self.logit_head(self.trunk(self._ws(states)))

    def dequantize(self, states: Tensor, idx: Tensor) -> Tensor:
        """(state, code index) -> an action vector in the ORIGINAL space.

        Without the residual head this returns the bare centroid, which is a
        coarse role code; with it, the state supplies the problem-specific
        part.  Both are available so the ablation is a flag, not a rewrite.
        """
        base = self.codebook[idx]
        if self.residual_head is not None:
            h = self.trunk(self._ws(states)) + self.code_emb(idx)
            base = base + self.residual_head(h)
        return base * self.a_std + self.a_mean

    def loss(self, states: Tensor, actions: Tensor,
             ctx: Tensor | None = None,
             ctx_mask: Tensor | None = None) -> dict[str, Tensor]:
        """CE on the prior + reconstruction of the observed action vector."""
        idx = self.quantize(actions)
        ce = nn.functional.cross_entropy(self.logits(states, ctx, ctx_mask),
                                         idx)
        out = {"ce": ce, "loss": ce}
        if self.residual_head is not None:
            rec = nn.functional.mse_loss(
                self.dequantize(states, idx), actions)
            out["recon"] = rec
            out["loss"] = ce + rec / float(actions.shape[-1])
        return out

    @torch.no_grad()
    def propose(self, states: Tensor, k: int, temperature: float = 1.0,
                sample: bool = False,
                generator: torch.Generator | None = None,
                ctx: Tensor | None = None,
                ctx_mask: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        """Top-``k`` (or sampled) code indices and their action vectors.

        Returns (idx [B, k], action vectors [B, k, D]).  Distinct indices are
        distinct BY CONSTRUCTION -- that is the property the generative
        proposer lacks.
        """
        lg = self.logits(states, ctx, ctx_mask) / max(temperature, 1e-6)
        if sample:
            probs = lg.softmax(-1)
            idx = torch.multinomial(probs, k, replacement=False,
                                    generator=generator)
        else:
            idx = lg.topk(k, dim=-1).indices
        B = states.shape[0]
        flat = self.dequantize(
            states.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1),
            idx.reshape(-1),
        )
        return idx, flat.reshape(B, k, -1)

    # ----------------------------------------------------------------- io
    def save(self, path) -> None:
        torch.save({"cfg": vars(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "CodePrior":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(CodePriorConfig(**blob["cfg"]))
        model.load_state_dict(blob["state"])
        return model
