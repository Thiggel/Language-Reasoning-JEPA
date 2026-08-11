"""Open-ended action proposal by CEM in the action-embedding space.

The planner receives no candidate menu and no catalogue.  At a state ``s`` it
searches the continuous action-embedding space directly: a population of action
codes ``u`` is scored by the *LDAD cycle*, i.e. the observed-action decoder
reconstructs a phrase from the predicted displacement
``predictor(s, u) - s``, and that phrase is re-encoded by the action encoder.
A code scores well when its decoded phrase is confident under the decoder and
re-embeds close to the code itself; the second term is the off-manifold
penalty that keeps the search on the manifold of codes the model can actually
express in language.  The surviving elites are decoded to phrases, and only
then does the environment ground them (with the single parser
:func:`textjepa.data.igsm.render.parse_action_phrase`) — text out, actions in,
never a menu in.

The decoding and scoring itself is *not* implemented here: it is the shared
LDAD cycle of :mod:`textjepa.planning.ldad_decode`, the same path
``ldad_cycle`` and ``generator_cycle`` use in ``planning.search``.  What is
specific to this module is the search over codes: the training-phrase prior,
the population loop, and the off-manifold penalty.

The initialization Gaussian is fitted on action phrases of *training* problems
only, so nothing about the evaluation problems' action spaces leaks into the
proposal distribution.  Everything here runs under ``torch.no_grad()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import parse_action_phrase
from textjepa.planning.ldad_decode import (
    delta_logits,
    encode_phrases,
    greedy_phrases,
    require_ldad_decoder,
    training_action_codes,
)

Tensor = torch.Tensor

MIN_STD = 1e-3


@dataclass
class ActionPrior:
    """Diagonal Gaussian over training-problem action embeddings.

    ``scale`` is the mean L2 norm of the codes it was fitted on. It is the
    unit in which off-manifold distances are measured, so that the penalty
    weight means the same thing across checkpoints whose action embeddings
    happen to live at different magnitudes.
    """

    mean: Tensor
    std: Tensor
    scale: Tensor | float = 1.0

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        noise = torch.randn(
            n, self.mean.shape[-1], device=self.mean.device,
            dtype=self.mean.dtype, generator=generator,
        )
        return self.mean + noise * self.std


def diagonal_prior(codes: Tensor) -> ActionPrior:
    if codes.ndim != 2 or len(codes) == 0:
        raise ValueError("codes must be a nonempty [n, d_action] tensor")
    std = (
        codes.std(0, unbiased=False)
        if len(codes) > 1
        else torch.ones_like(codes[0])
    )
    scale = codes.norm(dim=-1).mean().clamp_min(MIN_STD)
    return ActionPrior(codes.mean(0), std.clamp_min(MIN_STD), scale)


def cem_gaussian(
    score_fn: Callable[[Tensor], Tensor],
    prior: ActionPrior,
    population: int = 64,
    elites: int = 8,
    iterations: int = 3,
    prior_anchor: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Maximize ``score_fn`` with diagonal-Gaussian CEM anchored to ``prior``.

    Returns the final elite population and its scores, best first.  The
    refitted statistics are mixed with the prior statistics at weight
    ``prior_anchor``, which stops the population from collapsing onto an
    off-manifold spike of the learned score.
    """
    if population < 1 or elites < 1 or iterations < 1:
        raise ValueError("population, elites, and iterations must be positive")
    if not 0.0 <= prior_anchor <= 1.0:
        raise ValueError("prior_anchor must lie in [0, 1]")
    keep = min(elites, population)
    mean, std = prior.mean, prior.std
    best: Tensor | None = None
    best_scores: Tensor | None = None
    for _ in range(iterations):
        candidates = ActionPrior(mean, std).sample(population, generator)
        scores = score_fn(candidates)
        order = torch.argsort(scores, descending=True, stable=True)[:keep]
        best, best_scores = candidates[order], scores[order]
        fit_mean = best.mean(0)
        fit_std = (
            best.std(0, unbiased=False)
            if len(best) > 1
            else torch.zeros_like(fit_mean)
        )
        mean = (1.0 - prior_anchor) * fit_mean + prior_anchor * prior.mean
        std = (
            (1.0 - prior_anchor) * fit_std + prior_anchor * prior.std
        ).clamp_min(MIN_STD)
    return best, best_scores


class CEMCycleProposer:
    """Propose action phrases at a state without any candidate interface."""

    def __init__(
        self,
        model,
        vocab,
        device: torch.device,
        population: int = 64,
        elites: int = 8,
        iterations: int = 3,
        offmanifold_lambda: float = 1.0,
        prior_anchor: float = 0.1,
    ):
        require_ldad_decoder(model, "cem_cycle proposal")
        self.model = model
        self.vocab = vocab
        self.device = device
        self.population = int(population)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.offmanifold_lambda = float(offmanifold_lambda)
        self.prior_anchor = float(prior_anchor)
        self.prior: ActionPrior | None = None

    # ---------------------------------------------------------------- prior
    @torch.no_grad()
    def fit_prior(self, problems: list[Problem]) -> ActionPrior:
        """Fit the initialization Gaussian on training-problem phrases."""
        self.prior = diagonal_prior(
            training_action_codes(self.model, self.vocab, self.device, problems)
        )
        return self.prior

    def _require_prior(self) -> ActionPrior:
        if self.prior is None:
            raise RuntimeError(
                "fit_prior must be called with training problems before "
                "proposing (the evaluation problems' phrases must not be used)"
            )
        return self.prior

    def _encode_phrases(self, phrases: list[str]) -> Tensor:
        """[n] phrases -> [n, d_action] action codes (shared encode path)."""
        return encode_phrases(self.model, self.vocab, self.device, phrases)

    # --------------------------------------------------------------- decode
    @torch.no_grad()
    def cycle_scores(
        self, state: Tensor, codes: Tensor
    ) -> tuple[list[str], Tensor]:
        """Decode each code's phrase and score it with the off-manifold penalty.

        The decode-and-score step is the shared LDAD cycle
        (:func:`ldad_decode.greedy_phrases`); what this method adds is the
        penalty for codes whose own phrase does not re-embed back onto them.
        Distances are expressed in units of the prior's mean embedding norm,
        so ``offmanifold_lambda`` is comparable across checkpoints.
        """
        prior = self._require_prior()
        phrases, likelihood = greedy_phrases(
            delta_logits(self.model, state, codes), self.vocab
        )
        recoded = self._encode_phrases(phrases)
        offmanifold = (codes - recoded).square().sum(-1) / prior.scale**2
        return phrases, likelihood - self.offmanifold_lambda * offmanifold

    # -------------------------------------------------------------- propose
    @torch.no_grad()
    def propose(
        self,
        state: Tensor,
        problem: Problem,
        executed: frozenset[int],
        top_k: int = 0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], list[int], dict[str, float]]:
        """Return kept candidates, all parsed actions, and step diagnostics.

        ``executed`` masks the planner's OWN executed (or beam-imagined)
        actions, exactly as in ``ldad_cycle``: an already-resolved variable
        stays "computable" as far as the decoder is concerned, so without the
        mask the planner loops on no-op re-proposals. No oracle is consulted.
        """
        self._require_prior()
        codes, _ = cem_gaussian(
            lambda batch: self.cycle_scores(state, batch)[1],
            self.prior,
            population=self.population,
            elites=self.elites,
            iterations=self.iterations,
            prior_anchor=self.prior_anchor,
            generator=generator,
        )
        phrases, scores = self.cycle_scores(state, codes)
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        actions: list[int] = []
        parsed: list[int] = []
        n_parseable = 0
        for index in order:
            action = parse_action_phrase(problem, phrases[index])
            if action is None:
                continue
            n_parseable += 1
            if action not in parsed:
                parsed.append(action)
            if action in executed or action in actions:
                continue
            actions.append(action)
        if top_k > 0:
            actions = actions[:top_k]
        diagnostics = {
            "n_proposed": float(len(phrases)),
            "n_parseable": float(n_parseable),
            "n_unparseable": float(len(phrases) - n_parseable),
            "n_unique": float(len(parsed)),
            "n_kept": float(len(actions)),
            "parse_rate": n_parseable / max(len(phrases), 1),
        }
        return actions, parsed, diagnostics
