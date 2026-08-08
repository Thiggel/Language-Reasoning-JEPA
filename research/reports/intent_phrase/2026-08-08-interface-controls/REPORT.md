# Interface controls: what happens without the feasible-action menu

_2026-08-08. Round dirs: `2026-08-07-intent-interface-controls-v1`,
`2026-08-07-intent-action-prior-v1`. All evals: 300 val episodes,
root-balanced beam B=8, slack-curve evaluator (numbers below are strict
success at depths 1/2/4/8/16 and mean invalid-action rate)._

## Question (owner directive 2026-08-07)

The headline protocol gives every planner the environment's feasible-action
menu at depth 1 and symbolic future menus (candidate-privileged) at deeper
lookahead. Does anything work without it? And can a learned prior over
action embeddings — an MLP predicting a Gaussian over the sentence-embedding
space of actions — replace the menu entirely?

## Answer so far

Without the menu, EVERY method collapses to ~0 strict success at every
depth. The collapse is uniform across methods, so the with-menu comparison
table remains fair. The failure is specifically missing feasibility
knowledge, not missing action preference: the learned Gaussian prior ranks
the demonstrator's action ~3.6 of ~9 catalogue entries on average, but
proposals that are not yet computable (unresolved prerequisites) burn the
step budget (invalid=noop) 73–100% of the time.

## Results (strict D1/D2/D4/D8/D16; invalid rate range)

Full catalogue at every depth (no feasibility oracle, no symbolic menus,
invalid=noop):

| cell | strict | invalid |
|---|---|---|
| recipe s0, eval full-catalogue | .000/.003/.000/.000/.000 | .76–1.0 |
| TD-JEPA s0, eval full-catalogue | all .000 | .94–1.0 |
| GoalHead s0, eval full-catalogue | all .000 | .74–.99 |
| recipe RETRAINED full-catalogue (`mix4-fullcat-s0`) | .057/.000/.000/.000/.000 | .77–1.0 |

Learned prior (`mix4-prior-s0`: recipe + Gaussian action prior p(u(a)|s),
NLL on observed actions, + pairwise feasibility head; planning from
prior-filtered catalogue only):

| eval | strict | invalid |
|---|---|---|
| top-8 | .000 everywhere (slack-4 ≤ .003) | ~.87 |
| top-8 + pairwise-feasibility gate | identical to ungated | ~.87 |
| top-3 (gated or not — identical) | ~.003 | ~.73 |
| top-2 + gate | .007–.013 | ~.72 |

Prior health: `mean_prior_rank` (1-based rank of a correct feasible
necessary action under the prior; oracle label used for measurement only)
= 3.5–3.7 at all depths.

## Diagnosis

The pairwise feasibility head (MLP on [pooled state, candidate embedding])
plateaus at 58% accuracy with near-constant logits (positive/negative
logit gap ~0.3; measured spread across candidates ~1e-4), so gating with it
is a no-op. This is structural: iGSM feasibility is relational — an action
is executable iff ALL its prerequisite variables are already resolved —
and that predicate is not linearly readable from a pooled state/action
pair, but is directly expressible by attention over the executed-action
history.

## Fix under test

Commit `1321619`: the history-attention support head (candidates attend
over executed — or beam-imagined — action codes; learned, no oracle) wired
into the intent track: causal per-prefix history in training supervision,
history threaded through the planner's prior gate, correct action
dimensionality. Cell `mix4-prior-hist-s0-v1` (recipe + prior +
history-attention feasibility head, fully menu-free eval, gate + top-8)
running on gruenau1 GPU2. Outcome decides whether the paper's menu-free
protocol is "learned prior + learned feasibility works" or an honestly
reported limitation backed by the uniform-collapse table above.

## Evidence labels

Feasible menu at D1 = environment interface (not privileged). Symbolic
future menus at D>1 = candidate-privileged (labelled throughout the
paper). Full-catalogue and learned_catalogue protocols use no oracle at
plan time; `mean_prior_rank` uses oracle labels for measurement only.

## Final result (2026-08-08, after the fix)

The `mix4-prior-hist-s0-v1` failure above had a second, separate cause:
the recipe experiment config sets `value_detach=false`, so the prior NLL
gradients reached the action encoder and collapsed all intent-phrase
embeddings within one epoch (action_std 2.4 -> 0.002) — every candidate
became indistinguishable, which also explains the constant feasibility
logits at exactly the majority-class rate. Fix `d2f1cf5`: prior and
feasibility supervision always detach their inputs (read-outs, never
shapers).

The rerun `mix4-prior-hist-s0-v2` trained with healthy embeddings
(action_std 7.5) and STILL fails menu-free: strict ~0 at all depths,
invalid 72–82%, feasibility accuracy 52% (chance), prior rank ~4.3 of ~9.

**Conclusion.** Menu-free planning fails for every method, and after three
training iterations (pairwise head; history-attention head; collapse fix)
the reason is clear: deciding whether an action is executable requires the
candidate's prerequisite list, which is stated in the prompt text, and
nothing in JEPA training forces that relational structure into the pooled
state. The paper reports the interface axis as: headline protocol with the
environment menu (fair — collapse without it is uniform across methods),
plus this section as the documented limitation with the collapse control.
