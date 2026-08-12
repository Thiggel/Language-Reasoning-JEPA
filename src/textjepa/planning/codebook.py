"""Open-ended action proposal from a learned discrete action codebook.

Like ``cem_cycle`` this interface shows the planner no candidate menu and no
per-problem catalogue: at a state ``s`` it only ever looks at a fixed set of
``K`` action codes, fitted once at evaluation time by k-means over the action
embeddings of *training* problems.  No retraining is involved -- the codebook
is a summary of where the checkpoint's own action embeddings already live.
Empirically those embeddings fall into a handful of tight clusters (one per
operation, plus the leaf lookups), so a small codebook covers the space the
model can express.

Each code is scored by the same LDAD cycle every other open-ended interface
uses (:mod:`textjepa.planning.ldad_decode`): the predictor imagines the
displacement ``predictor(s, u) - s``, the observed-action decoder greedily
reads a phrase out of it, and the code's score is that phrase's own mean token
log-probability.  Only then is the phrase grounded by the single parser
:func:`textjepa.data.igsm.render.parse_action_phrase` -- text out, actions in,
never a menu in.

Unlike the CEM the proposal is deterministic: at a given state the same ``K``
codes are always scored, so nothing here consumes RNG.  Everything runs under
``torch.no_grad()``.
"""

from __future__ import annotations

import torch

from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import parse_action_phrase
from textjepa.planning.ldad_decode import (
    delta_logits,
    greedy_phrases,
    require_ldad_decoder,
    training_action_codes,
)

Tensor = torch.Tensor

KMEANS_ITERS = 50
# Centroid movement below which Lloyd's iteration is considered converged.
KMEANS_TOL = 1e-6


def _seeded_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _kmeans_plus_plus(
    embeddings: Tensor, k: int, generator: torch.Generator
) -> Tensor:
    """k-means++ initialization: spread the first centres by squared distance."""
    n = embeddings.shape[0]
    first = torch.randint(
        n, (1,), generator=generator, device=embeddings.device
    ).item()
    centres = [embeddings[first]]
    closest = (embeddings - centres[0]).square().sum(-1)
    for _ in range(1, k):
        weights = closest.clamp_min(0.0)
        if float(weights.sum()) <= 0.0:
            # Fewer distinct points than centres: fall back to a uniform draw
            # so the codebook still has K rows (duplicates are harmless).
            index = torch.randint(
                n, (1,), generator=generator, device=embeddings.device
            ).item()
        else:
            index = int(
                torch.multinomial(weights, 1, generator=generator).item()
            )
        centres.append(embeddings[index])
        closest = torch.minimum(
            closest, (embeddings - centres[-1]).square().sum(-1)
        )
    return torch.stack(centres)


@torch.no_grad()
def fit_codebook(embeddings: Tensor, k: int = 64, seed: int = 0) -> Tensor:
    """k-means over action embeddings -> a [K, d] codebook.

    Deterministic given ``seed`` (k-means++ init from a seeded generator, then
    Lloyd's iterations, which are themselves deterministic).  Empty clusters
    keep their previous centre rather than being re-seeded, so the result does
    not depend on any ambient RNG.  ``k`` is clamped to the number of available
    embeddings.
    """
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must be a nonempty [n, d] tensor")
    if k < 1:
        raise ValueError("k must be positive")
    k = min(int(k), len(embeddings))
    centres = _kmeans_plus_plus(
        embeddings, k, _seeded_generator(seed, embeddings.device)
    )
    for _ in range(KMEANS_ITERS):
        assignment = torch.cdist(embeddings, centres).argmin(-1)
        updated = centres.clone()
        for cluster in range(k):
            members = embeddings[assignment == cluster]
            if len(members):
                updated[cluster] = members.mean(0)
        shift = (updated - centres).norm(dim=-1).max()
        centres = updated
        if float(shift) <= KMEANS_TOL:
            break
    return centres


class CodebookCycleProposer:
    """Propose action phrases from a fitted codebook, with no menu access."""

    def __init__(
        self,
        model,
        vocab,
        device: torch.device,
        k: int = 64,
        seed: int = 0,
    ):
        require_ldad_decoder(model, "codebook_cycle proposal")
        if int(k) < 1:
            raise ValueError("codebook_k must be positive")
        self.model = model
        self.vocab = vocab
        self.device = device
        self.k = int(k)
        self.seed = int(seed)
        self.codebook: Tensor | None = None

    # ------------------------------------------------------------- codebook
    @torch.no_grad()
    def fit_prior(self, problems: list[Problem]) -> Tensor:
        """Fit the codebook on TRAINING-problem action phrases only.

        Named ``fit_prior`` because it is the same hook the planner calls for
        every eval-time proposal distribution (see ``cem_cycle``); it uses the
        one shared embedding-collection path.
        """
        self.codebook = fit_codebook(
            training_action_codes(
                self.model, self.vocab, self.device, problems
            ),
            k=self.k,
            seed=self.seed,
        )
        return self.codebook

    def _require_codebook(self) -> Tensor:
        if self.codebook is None:
            raise RuntimeError(
                "fit_prior must be called with training problems before "
                "proposing (the evaluation problems' phrases must not be used)"
            )
        return self.codebook

    # --------------------------------------------------------------- decode
    @torch.no_grad()
    def cycle_scores(self, state: Tensor) -> tuple[list[str], Tensor]:
        """Decode and score every code of the codebook at ``state`` (batched)."""
        codes = self._require_codebook()
        return greedy_phrases(
            delta_logits(
                self.model, state, codes,
                context="candidate_interface=codebook_cycle",
            ),
            self.vocab,
        )

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
        actions, exactly as in ``ldad_cycle`` and ``cem_cycle``: an already
        resolved variable stays "computable" as far as the decoder is
        concerned, so without the mask the planner loops on no-op
        re-proposals.  No oracle is consulted.  ``generator`` is accepted for
        interface compatibility and unused: the proposal is deterministic.
        """
        phrases, scores = self.cycle_scores(state)
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


class CodebookGroundProposer(CodebookCycleProposer):
    """Codebook latents grounded environment-side by nearest neighbour.

    Same fitted codebook as ``codebook_cycle``, different grounding contract:
    instead of decoding each code to text and demanding an exactly parseable
    phrase (which fails because compute phrases are problem-specific
    compositions -- see the 2026-08-12 codebook diagnosis), the ENVIRONMENT
    matches each proposed latent against the embeddings of its own action
    inventory and executes the nearest action.  This mirrors how text-game
    environments (e.g. ALFWorld) fuzzy-match generated commands against their
    admissible-command inventory: the grounding is an environment-interface
    property.  The model side still never sees a menu, receives no
    feasibility information, and must place its proposal vectors well --
    candidates are then ranked by the usual LDAD cycle score of the grounded
    action's own phrase.  Evidence label: environment-side action grounding
    (weaker than feasible_menu -- no feasibility bit, no explicit list to
    score -- stronger than the exact-parse contract of *_cycle).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._catalogue_cache: dict[tuple[str, ...], Tensor] = {}

    @torch.no_grad()
    def _catalogue_embeddings(
        self, phrases: list[str]
    ) -> Tensor:
        from textjepa.planning.ldad_decode import encode_phrases

        key = tuple(phrases)
        hit = self._catalogue_cache.get(key)
        if hit is None:
            if len(self._catalogue_cache) > 8:
                self._catalogue_cache.clear()
            hit = encode_phrases(self.model, self.vocab, self.device, phrases)
            self._catalogue_cache[key] = hit
        return hit

    @torch.no_grad()
    def propose(
        self,
        state: Tensor,
        problem: Problem,
        executed: frozenset[int],
        top_k: int = 0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], list[int], dict[str, float]]:
        from textjepa.data.igsm.render import catalogue_phrases
        from textjepa.planning.ldad_decode import delta_logits, phrase_log_probs

        codes = self._require_codebook()
        phrases = catalogue_phrases(problem)
        u_cat = self._catalogue_embeddings(phrases)
        grounded_rows = torch.cdist(codes, u_cat).argmin(1).tolist()
        # Environment-side grounding: each catalogue row denotes one action.
        parsed: list[int] = []
        for row in grounded_rows:
            action = parse_action_phrase(problem, phrases[row])
            if action is not None and action not in parsed:
                parsed.append(action)
        # Rank the grounded (unique) candidates by the standard cycle score
        # of their own phrase at this state -- identical scoring rule to
        # ldad_cycle, applied to the codebook-grounded subset only.
        if parsed:
            token_ids = [
                self.vocab.encode(phrases[a]) for a in parsed
            ]
            scores = phrase_log_probs(
                delta_logits(
                    self.model, state,
                    u_cat[torch.tensor(parsed, device=u_cat.device)],
                    context="candidate_interface=codebook_ground",
                ),
                token_ids,
            )
            order = torch.argsort(scores, descending=True, stable=True)
            ranked = [parsed[i] for i in order.tolist()]
        else:
            ranked = []
        actions = [a for a in ranked if a not in executed]
        if top_k > 0:
            actions = actions[:top_k]
        diagnostics = {
            "n_proposed": float(len(codes)),
            "n_parseable": float(len(grounded_rows)),
            "n_unparseable": 0.0,
            "n_unique": float(len(parsed)),
            "n_kept": float(len(actions)),
            "parse_rate": 1.0 if len(codes) else 0.0,
        }
        return actions, parsed, diagnostics
