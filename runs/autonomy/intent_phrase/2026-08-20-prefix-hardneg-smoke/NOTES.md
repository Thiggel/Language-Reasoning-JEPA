# Harder negatives for energy_prefix_rank — design note

## Why

`energy_prefix_acc` reaches ~.99 by epoch 1. It starts at chance and takes
about an epoch, so it is learning something real (unlike the length-cue bug,
which saturated in 40 steps). But near-ceiling accuracy means the gradient
dies early, which would fit a "training accuracy great, depth still doesn't
help" outcome.

## What the coordinator asked for vs what is actually available

**Requested #1, "legal-but-useless insertion" (insert an intent that IS
feasible but is not the one the environment took).** NOT implementable here
as a clean label, and I did not build it. In `SymbolicEnv`, a rollout
continuation is drawn UNIFORMLY AT RANDOM from `feasible_actions()`
(`nxt = feasible[rng.randrange(len(feasible))]`, both datasets). So at
rollout depth >= 1 a feasible-but-not-taken action is statistically
indistinguishable from the taken one — neither is better, the sampler just
picked one. Training the head to rank the taken one below an untaken feasible
one would be fitting the sampler's coin flip, i.e. pure label noise pushing
the term back toward 50%.

What IS available, and what I built: the dataset's existing
`rollout_counterfactual_k` sampler already draws half its negatives from
ALREADY-RESOLVED variables. Those are not literally legal — `feasible_actions`
excludes resolved variables, so they execute as no-ops like the premature ones
— but they are *legal-looking*: the intent re-derives a fact the state already
contains, so it reads as a perfectly sensible sentence. Separating that from a
step that made progress is much closer to the useful/pointless judgement than
separating it from a premature intent whose parents are still unknown.

Honest label: this is "legal-LOOKING but pointless", not "legal but useless".

**The principled route to a genuinely legal-but-useless negative** (not built,
noted for later): make the rollout follow the remaining true solution trace
instead of random feasible actions. Then "the continuation that occurred" is
the actual solution, and feasible alternatives at each depth are genuinely
off-path — exactly the depth-0 contrast `energy_cf_feasibility_rank` already
uses at the anchor, extended to depth >= 1. That is a dataset-level change to
the rollout policy, not a sampler tweak.

**Requested #2, deeper insertion.** Built as `depth_bias: late`, weighting the
insertion depth linearly with depth. Measured effect on the current data:
mean insertion depth 2.079 (uniform) -> 2.349 (late). The effect is modest
because many rollouts are short (horizon is sampled from [1,2,4,8] and a
rollout stops once solved).

## Switches (both default to current behaviour)

    objective.energy_prefix_rank:
      cf_kind:    all (default) | premature (easy control) | resolved (hard)
      depth_bias: uniform (default) | late

Independently switchable so the two can be attributed separately. The dataset
now tags each rollout counterfactual (1 = premature, 2 = already resolved);
this selects WHICH negatives are offered and is never a ranking label — the
label stays "which continuation actually occurred". Same category as the
existing `invalid_counterfactual_resolved_k` knob in the frozen recipe.

## Regression evidence

Every pre-existing collated tensor is bit-identical between snapshot c8115c9
and the new code on the same seed (only new key: `ga_roll_cf_kind`). Length
matching asserted for all five variants. Gradients reach encoder, predictor
and Energy head in all of them.

## RESULT (2026-08-20): the difficulty ordering is the OPPOSITE of the premise

Four 120-step runs, identical seed and init, iGSM-med recipe (ecf16 +
energy_prefix_rank w=4), microbatch 4 x accum 4 on gruenau2 RTX 6000s.
Mean `energy_prefix_acc` over the last 5 logged steps (80-120); LOWER = HARDER:

| cf_kind / depth_bias | mean acc @80-120 | reading |
|---|---|---|
| premature / uniform  | **.507** | HARDEST |
| all / uniform (default) | .609 | reference |
| all / late           | .613 | no effect |
| resolved / uniform   | .710 | EASIEST |

1. **`resolved` is EASIER, not harder — the hypothesis is refuted.** Inserting
   an already-resolved intent was predicted to be the hard case because it
   reads as a sensible sentence. Empirically it is the easiest of the three.
   Most likely because the state already contains that fact, so the imagined
   next state barely moves: a very distinctive geometric signature. Do NOT
   launch `cf_kind=resolved` as the harder-negative arm.

2. **`premature` IS the harder negative** — the switch I built as the "easy
   control" turned out to be the hard one. It is ~10 points below the default
   and its curve flattens after step 80 (.553 -> .464 -> .437 -> .467) while
   the others keep climbing. That is exactly the "starts at chance and climbs
   more slowly" signature we wanted. A premature intent references variables
   whose parents are not resolved yet, so telling it apart requires tracking
   what is currently known -- plausibly the judgement planning needs.

3. **`depth_bias=late` does nothing measurable** (.613 vs .609). It only moved
   mean insertion depth 2.05 -> 2.28 because most rollouts are short. Not
   worth a training arm on this evidence.

RECOMMENDATION if epoch 2/4 shows depth still failing: launch ONE arm with
`cf_kind=premature, depth_bias=uniform`. Drop `resolved` and drop the depth
bias. If that is still not enough, go to the rollout-policy change described
above (follow the remaining true solution trace), which is the only route to a
genuinely legal-but-useless negative.

CAVEAT: 120 steps, one seed. The within-run swing at fixed step is +-.10, so
the premature/resolved separation (.507 vs .710) is well outside it but the
all vs all_late difference (.609 vs .613) is not resolvable.
