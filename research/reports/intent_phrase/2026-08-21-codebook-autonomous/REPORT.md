# The autonomous codebook interface: gate passed, planning ladder .115 → .285, coverage is the open gap

Date: 2026-08-21 · faithful iGSM-med · rounds
`runs/autonomy/intent_phrase/2026-08-20-action-decoder-gate-v1/` and
`2026-08-21-code-prior-v1/` (both carry HANDOFF.md files with artifact paths
and continuation commands).

## The target architecture (owner's directive, in force)

A learned **codebook of actions**; the planner proposes and rolls out in
latent space from that codebook **without ever seeing a menu, catalogue, or
feasible set**; a **detached decoder** (no shared tensor with the backbone,
tested) renders the planned states/actions into text; the environment is used
only to execute the decoded text. `prior_propose` (token-head) is a reference,
not the target; `codebook_ground` (snaps onto the problem's own action list)
is catalogue-privileged and demoted to a labeled diagnostic.

## Gate: can a detached decoder render held-out action phrases? — PASSED

Context-conditioned detached decoder, trained on demonstrated traces only
(commit range 51859f6…c5ec3ae): **.9978 exact match** over 1833 val states,
**.9992 on the 1219 phrases never seen verbatim in training** (novel beats
seen → composition, not memorisation). Capacity-matched ablations, identical
init/batches:

| arm | exact | novel |
|---|---|---|
| full (action vector + context) | .998 | .999 |
| context only | .446 | .373 |
| action vector only (= old failure mode) | .432 | .279 |
| shuffled action, real context | .469 | .391 |

The action vector carries role/operation but not names; the context carries
the names; cross-attention composes them. This dissolves the obstacle that
gave every previous catalogue-free proposer a .000 parse rate.

## End-to-end planning ladder (d1, 200 episodes ID, success at cap 1.0 / 1.25)

All rows use the answer-emission-compatible protocol; `rescore_budget.py`
reproduces every cap column from the saved episodes.

| step | change | true-next recall | plan cap 1.0 / 1.25 |
|---|---|---|---|
| 1 | MLP prior, K=64 codes, 16 proposals | .586 | .115 / .165 |
| 2 | + 64 proposals ("wide-K") | .715 | .215 / .290 |
| 3 | ecf16-geometry retrain (hypothesis REFUTED) | .717 | .215 / .345 |
| 4 | capacity/epoch sweep | flat (~.72 top4) | – |
| 5 | K=32 coarse codes (top4 .830 but ambiguous decode) | .555 | .080 / .145 |
| 6 | **context-conditioned prior** (commit 001c3ca) | **.766** | **.285 / .395** |

References on the same checkpoints: token-head `prior_propose` **.840 / .895**;
true-oracle proposer ceiling **.945** with every miss being proposal
exhaustion (see `2026-08-21-true-oracle-ladder/`); `codebook_ground`
collapsed from .890 (ecf16 ckpt) to .010 (roll124 ckpt) — codebook snapping is
both privileged and fragile, another reason it is not the story.

Step 3 killed the "it's the checkpoint geometry" hypothesis: decoder+prior
retrained on the checkpoint whose geometry supported .890 snapped planning
gave identical recall. Step 5 mapped the K-axis as a pure
predictability-vs-ambiguity trade-off with no good operating point. Step 6
(the context prior: the pooled state cross-attends over the frozen encoder's
context hiddens, 2-layer decoder pattern, backwards-compatible loader) is the
same medicine that fixed the decoder, applied to the prior: top-4 code recall
.720→.850, val top-1 .378→.529.

## Current leak and the round in flight (`offpath-v1`)

At the context-prior operating point, **39% of decoded proposals fail to
parse** — decoder and prior were trained only on on-path states, and planning
visits off-path states constantly (the risk flagged in the gate handoff). New
`--offpath-prob` in `scripts/train_action_decoder.py`: the cache walk steps
onto a random feasible action with probability p (label is still just the
action taken; self-supervised). Status:

- Decoder retrained on offpath 0.3 cache: **gate holds at .997 exact
  (.995 novel)** — off-path decoding is solved.
- Context prior retraining on the same cache; two infra bugs found and fixed
  on the way: torch's fused `_transformer_encoder_layer_fwd` fast path raises
  an illegal memory access on these context batches (fix: disable the MHA
  fast path in `train_code_prior.py`, committed), then OOM from the up-to-4×
  longer off-path histories (fix: batch 16). Third attempt running on
  gruenau12:4 with bench + d1 planning chained.

## Decision frame

The proposer ceiling (.945) minus the current autonomous point (.285) is the
whole remaining story, and it decomposes measurably: code recall (.85 top-4)
× decode parse (.61, off-path fix in flight) × per-step compounding. If
off-path exposure lifts parse to ~1 and recall holds, the compounded per-step
success supports planning in the .5–.6 range; beyond that the prior itself
needs more capacity per state (next lever: larger context prior / longer
training on more cached problems), not a different architecture. The
architecture question the owner posed is answered: **yes — a learned codebook
+ detached decoder can plan without ever seeing a menu, and every component
gate has passed; the remaining gap is proposal coverage, which is now a single
measurable number.**
