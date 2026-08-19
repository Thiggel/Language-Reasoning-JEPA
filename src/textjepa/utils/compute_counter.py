"""Test-time-compute accounting shared by the LM baselines and the JEPA planner.

The paper needs one x-axis on which "more test-time compute" is comparable
across two very different families:

* autoregressive LM baselines, whose compute is dominated by generated tokens
  (one full transformer forward per generated token in this codebase — there
  is no KV cache), plus any extra scoring/looped passes;
* the JEPA planner, whose compute is a mixture of backbone (encoder) forwards
  over token sequences and much cheaper latent predictor / energy forwards.

Counting raw "forwards" is not comparable between the two (a JEPA predictor
forward is a two-layer MLP; an LM forward is a 12-layer transformer over the
whole prefix).  So we record three things and let the paper plot whichever it
argues for:

``backbone_forwards``
    number of calls into the *token transformer* (LM blocks / JEPA encoder),
    counting a batched call as ``batch`` forwards.
``backbone_token_positions``
    sum over those calls of ``batch * sequence_length``.  This is the
    FLOP-proportional quantity and is the recommended comparable x-axis: an
    LM that generates 500 tokens over a 300-token prefix pays ~500*400
    positions, and a JEPA planner that encodes 6 prefixes and one batched
    candidate pass pays what it actually pays.
``generated_tokens``
    tokens produced by autoregressive decoding (LM solution tokens, JEPA
    proposer phrases / model-written outcomes).  The "tokens" axis reviewers
    expect for test-time-compute plots.

``latent_forwards`` / ``latent_slots`` track the JEPA-side predictor and
energy calls separately, so we can state explicitly that latent search is
cheap and is *not* what the token axis is measuring.

Nothing here changes model behaviour; a counter is a passive accumulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class ComputeCounter:
    backbone_forwards: int = 0
    backbone_token_positions: int = 0
    generated_tokens: int = 0
    latent_forwards: int = 0
    latent_slots: int = 0
    episodes: int = 0
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------ recording
    def backbone(self, batch: int, seq_len: int, generated: int = 0) -> None:
        """One (possibly batched) forward of the token transformer."""
        self.backbone_forwards += int(batch)
        self.backbone_token_positions += int(batch) * int(seq_len)
        self.generated_tokens += int(generated)

    def latent(self, batch: int, slots: int = 1) -> None:
        """One (possibly batched) predictor / energy forward in latent space."""
        self.latent_forwards += int(batch)
        self.latent_slots += int(batch) * int(slots)

    def bump(self, key: str, amount: int = 1) -> None:
        self.extra[key] = self.extra.get(key, 0) + int(amount)

    def new_episode(self) -> None:
        self.episodes += 1

    # ------------------------------------------------------------- reporting
    def totals(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d

    def per_episode(self) -> dict:
        n = max(self.episodes, 1)
        return {
            f"{k}_per_episode": v / n
            for k, v in self.totals().items()
            if k != "episodes"
        }

    def report(self) -> dict:
        out = {"totals": self.totals(), "per_episode": self.per_episode()}
        return out

    def __str__(self) -> str:  # pragma: no cover - debug convenience
        return json.dumps(self.report(), indent=2)
