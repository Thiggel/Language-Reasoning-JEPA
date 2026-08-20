"""Conditional density over next-action embeddings: p(a | s).

Motivation
----------
``FlatPlanner``'s menu-free proposer (``prior_propose``) samples K intent
phrases from the backbone's own next-token head and keeps the ones whose text
happens to match an action of the current problem.  That is random search
under a token-level prior, not a *model of the distribution over next
actions*.  This module fits that distribution directly, in the space the
planner actually scores in: the contextual action code
``model.encode_candidates_in_context(history, [phrase])``.

Why a coupling flow and not flow matching
-----------------------------------------
Flow matching gives cheap samples but only ODE-integrated (expensive,
approximate) likelihoods.  The decoding problem in this environment (see
``research/reports/intent_phrase/2026-08-12-codebook-diagnosis``) makes the
*density* the more valuable half of the model: a sampled embedding cannot be
turned back into executable text reliably, but a density can rerank text that
was produced by a prompt-grounded token decoder.  A RealNVP-style conditional
coupling flow gives exact log-density AND exact sampling in one model, so it
supports both the reranking arm and the direct-decode arm.  If the decode arm
ever works, the same object can be swapped for a flow-matching field with no
change to the planner interface.

Self-supervision
----------------
Training targets are the actions that ACTUALLY OCCURRED in the demonstrated
solution traces, encoded by the frozen checkpoint.  No symbolic labels, no
steps-to-go, no feasibility oracle (cf. the no-symbolic-heads rule in
CLAUDE.md).  "Which action occurred" is in the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

Tensor = torch.Tensor


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_hidden), nn.SiLU(),
        nn.Linear(d_hidden, d_out),
    )
    # Zero-init the last layer so every coupling starts as the identity map:
    # the flow begins exactly at the whitened Gaussian base density.
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class _Coupling(nn.Module):
    """Affine coupling: half the dims are passed through, half are rescaled.

    ``mask`` is 1 on the pass-through dims.  The scale/shift for the
    transformed half is produced from (masked input, condition), so the
    Jacobian stays triangular and its log-determinant is a plain sum.
    """

    def __init__(self, dim: int, cond_dim: int, hidden: int, flip: bool):
        super().__init__()
        mask = torch.zeros(dim)
        mask[: dim // 2] = 1.0
        if flip:
            mask = 1.0 - mask
        self.register_buffer("mask", mask)
        self.net = _mlp(dim + cond_dim, hidden, 2 * dim)
        # Bounds the affine scale; keeps the flow numerically stable at 768-d.
        self.scale_cap = 3.0

    def _params(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(torch.cat([x * self.mask, cond], -1))
        s, t = h.chunk(2, -1)
        s = self.scale_cap * torch.tanh(s / self.scale_cap)
        keep = 1.0 - self.mask
        return s * keep, t * keep

    def forward(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Data -> base.  Returns (y, log|det J|)."""
        s, t = self._params(x, cond)
        return x * torch.exp(-s) - t * torch.exp(-s), -s.sum(-1)

    def inverse(self, y: Tensor, cond: Tensor) -> Tensor:
        """Base -> data."""
        s, t = self._params(y, cond)
        return (y + t) * torch.exp(s)


@dataclass
class FlowPriorConfig:
    dim: int
    n_layers: int = 8
    hidden: int = 1024
    cond_dim: int = 256


class ConditionalFlowPrior(nn.Module):
    """p(action code | state), as an exact-likelihood conditional flow.

    Whitening statistics are part of the model: the raw contextual action
    codes of a 768-d transformer have wildly different per-dimension scales,
    and an unwhitened flow spends all its capacity on that.
    """

    def __init__(self, cfg: FlowPriorConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.register_buffer("a_mean", torch.zeros(d))
        self.register_buffer("a_std", torch.ones(d))
        self.register_buffer("s_mean", torch.zeros(d))
        self.register_buffer("s_std", torch.ones(d))
        self.cond = nn.Sequential(
            nn.Linear(d, cfg.cond_dim), nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        self.layers = nn.ModuleList(
            _Coupling(d, cfg.cond_dim, cfg.hidden, flip=bool(i % 2))
            for i in range(cfg.n_layers)
        )

    # ------------------------------------------------------------ scaling
    @torch.no_grad()
    def fit_scaling(self, actions: Tensor, states: Tensor) -> None:
        self.a_mean.copy_(actions.mean(0))
        self.a_std.copy_(actions.std(0).clamp_min(1e-3))
        self.s_mean.copy_(states.mean(0))
        self.s_std.copy_(states.std(0).clamp_min(1e-3))

    def _whiten(self, a: Tensor) -> Tensor:
        return (a - self.a_mean) / self.a_std

    def _unwhiten(self, x: Tensor) -> Tensor:
        return x * self.a_std + self.a_mean

    def _cond(self, states: Tensor) -> Tensor:
        return self.cond((states - self.s_mean) / self.s_std)

    # ------------------------------------------------------------ density
    def log_prob(self, actions: Tensor, states: Tensor) -> Tensor:
        """Exact log p(a | s), per row, in whitened action units.

        The constant offset from whitening (``-sum log a_std``) is included so
        the number is a genuine density in the raw code space and is
        comparable across checkpoints of the same width.
        """
        cond = self._cond(states)
        x = self._whiten(actions)
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer(x, cond)
            logdet = logdet + ld
        base = -0.5 * (x.square().sum(-1) + x.shape[-1] * torch.log(
            torch.tensor(2 * torch.pi, device=x.device, dtype=x.dtype)))
        return base + logdet - torch.log(self.a_std).sum()

    # ----------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, states: Tensor, n: int, generator=None,
               temperature: float = 1.0) -> Tensor:
        """``n`` action codes per row of ``states`` -> [B, n, D] raw codes."""
        b, d = states.shape
        cond = self._cond(states).repeat_interleave(n, 0)
        y = torch.randn(b * n, d, device=states.device, dtype=states.dtype,
                        generator=generator) * temperature
        for layer in reversed(self.layers):
            y = layer.inverse(y, cond)
        return self._unwhiten(y).view(b, n, d)

    # ------------------------------------------------------------ persist
    def save(self, path) -> None:
        torch.save({"cfg": vars(self.cfg), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "ConditionalFlowPrior":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(FlowPriorConfig(**blob["cfg"]))
        model.load_state_dict(blob["state"])
        return model.eval()


def maxmin_diverse(codes: Tensor, scores: Tensor, k: int,
                   pool_mult: float = 2.0) -> list[int]:
    """Pick ``k`` rows that are high-density AND mutually spread out.

    Two stages, deliberately, rather than one weighted sum: a weighted sum
    needs an arbitrary exchange rate between "log-density" and "distance", and
    whichever rate you choose, a tight high-density cluster wins every slot --
    which is exactly the collapse the current temperature-only proposer
    already suffers (~1.6 unique parseable intents per state).

    Stage 1 uses the learned density as a QUALITY GATE: keep the best
    ``pool_mult * k`` rows.  Stage 2 uses pure greedy max-min distance to fill
    the k slots from that shortlist, so every slot after the first is spent on
    something the earlier picks do not already cover.
    """
    n = codes.shape[0]
    if n <= k:
        return list(range(n))
    order = torch.argsort(scores, descending=True).tolist()
    pool = order[: max(k, min(n, int(round(pool_mult * k))))]
    idx = torch.tensor(pool, device=codes.device)
    sub = codes[idx].float()
    dist = torch.cdist(sub, sub)
    picked = [0]  # pool is score-sorted, so row 0 is the densest candidate
    closest = dist[0].clone()
    while len(picked) < k:
        util = closest.clone()
        util[torch.tensor(picked, device=util.device)] = float("-inf")
        nxt = int(util.argmax().item())
        picked.append(nxt)
        closest = torch.minimum(closest, dist[nxt])
    return [pool[i] for i in picked]
