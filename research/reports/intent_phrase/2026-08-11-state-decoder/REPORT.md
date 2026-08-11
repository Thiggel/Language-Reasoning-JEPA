# Frozen-state sentence decoder: what the latent states actually say

_2026-08-11. Owner-requested renderer: turn the sequence of latent states
into language, trained ONLY on the frozen backbone (JEPA-pure readout;
zero-gradient assertion in the trainer). Backbone: stab-ldad-ema-s0-v1
(LDAD headline recipe, seed 0). Decoder: 2-layer causal transformer, 1.19M
params, state-conditioned, 12 epochs / 20k problems per epoch (~25 min GPU).
Code: `src/textjepa/models/state_decoder.py`,
`scripts/{train,eval}_state_decoder.py`; run + tables:
`runs/autonomy/intent_phrase/2026-08-11-intent-state-decoder-v1/decoder-s0-v1/`.
500 held-out test problems throughout._

## Result 1 — TRUE encoded states are near-fully legible, numbers included

Decoding the encoder's state after each real step: 77% exact sentence
match, 96% token accuracy, 96% VALUE accuracy overall (per-step values stay
>= .91 at every depth; exact-match decline with step index is a
length/clause effect). A linear answer head on the final state reads the
final answer residue at 94% (chance 4%). The JEPA state is a faithful,
decodable record of the reasoning performed — including the arithmetic.

## Result 2 — IMAGINED states keep the discourse form, not the arithmetic

Rolling the predictor from s_0 along the true actions and decoding each
imagined state: token accuracy .92 at depth 1-2 but value accuracy only
.32, decaying to chance (1/23) by depth 5. The imagined state renders as a
plausible step sentence about a plausible variable with the wrong number.

## Controls (same decoder, same problems)

Prompt-only (decode step t from s_0, NO predictor): value acc .248 at step
1, token acc .561, exact .000 at every step. One-step-from-true-prefix
(one predictor application on the encoded true prefix): identical to
rollout at depth 1 by construction (.318/.923/.318), .030 exact / .046
value by step 3 — NOT better than the s_0 rollout at matched step.

Reading:
- Depth-1 imagined value accuracy is mostly prompt-guessable: one predictor
  step buys +7 points of value accuracy over reading the problem statement
  (.318 vs .248).
- What prediction genuinely contributes is STRUCTURE: token accuracy .561
  -> .923 with one predictor step; exact match 0 -> .318 (prompt-only exact
  is zero everywhere, so nonzero exact match is attributable to the
  predictor).
- Single-step fidelity, not compounded drift, is the bottleneck: fresh
  true-prefix one-step decoding is no better than the compounded rollout at
  matched depth. Wherever a step involves arithmetic, one application of
  the predictor already loses the number. (Step 1 looks best partly because
  it is disproportionately leaf lookups. Steps 7-8 have n<=60; ignore
  tails.)

## Paper story

Encoder states = complete decodable record (structure + values). Predictor
= propagates plan-relevant structure (which variable, which action, the
feasibility geometry that cycle-consistency reads at AUC .94) but not the
computation itself. Planning succeeds because the endpoint Energy needs
only the propagated structure/geometry; calibrated-value scores (TD-JEPA,
GoalHead) fail with depth because they need what the predictor does not
carry. This also gives the honest limitation sentence: the latent rollout
is a plan simulator, not an arithmetic simulator — with the frozen-state
decoder, executed plans (true states) render to fully correct worked
solutions, imagined ones to structurally correct previews.

Limitations: single seed/backbone; iGSM only; decoder capacity not swept
(a larger decoder could shift absolute levels, not the true-vs-imagined
contrast, but this is unverified).
