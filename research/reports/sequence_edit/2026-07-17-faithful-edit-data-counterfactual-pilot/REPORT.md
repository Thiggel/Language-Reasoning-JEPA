# How much edit experience should the model receive?

## The one-sentence answer

We do not yet know the useful amount: the previous model runs were invalid,
but the repaired data pipeline now passes its validity checks and a bounded
six-cell pilot can measure both unique-data and counterfactual-density effects.

## First, the idea in everyday language

Imagine teaching an editor by taking a correct worked solution, introducing a
handful of typos, and showing how to undo them. Seeing more independently
damaged solutions teaches breadth. At any damaged version, also asking “what
would this other edit produce?” teaches local action consequences. The open
question is whether breadth, alternatives, or both are needed before it makes
sense to test a more elaborate hierarchical editor.

The previous data loader accidentally glued a whole multi-step solution into
one enormous sentence. That made the first five jobs crash and, more
importantly, meant they were not testing the intended representation. The
loader now retains the official solution steps.

## Why this question matters

Hierarchy cannot rescue an editor that has too little varied experience or
cannot distinguish actions. Conversely, alternatives can waste memory or
teach an unrepresentative distribution. This pilot selects a defensible
primitive-data recipe before comparing flat, hierarchical, dense-rollout, and
action-decoding variants. It is a prerequisite, not evidence that the model
can autonomously edit or solve language problems.

## What we tested

The data audit used 128 official iGSM problems. Each problem supplies a prompt
and clean multi-step solution. A deterministic synthetic process inserts,
deletes, or replaces tokens, then records the exact reverse edit sequence.
The audit checked step preservation, exact recovery, edit composition, text
length, and distance from a minimum token-edit path. It also generated four
alternative edits per visited state to verify exact execution and count them.

The admitted one-seed GPU pilot has six independent cells:

| Cell | Unique train trajectories | Passes | Alternatives per state | Purpose |
|---|---:|---:|---:|---|
| H4 smoke | 128 | 1 | 0 | full-shape process check |
| flat-small | 512 | 3 | 0 | low-data anchor |
| flat-medium | 2,000 | 3 | 0 | shared comparison anchor |
| flat-base | 6,000 | 3 | 0 | initial data-curve endpoint |
| CF-1 | 2,000 | 3 | 1 | sparse alternative outcome |
| CF-4 | 2,000 | 3 | 4 | moderate alternative outcome |

## What a fair comparison means here

All counterfactual cells use the same uniform-local sampler, model, seed,
unique problem set, and counterfactual-loss weight. Alternatives are sampled
only from the observed current buffer and are mechanically executed. They do
not carry a preference, defect count, remaining distance, or target-relative
quality label. Thus K changes the number of exact outcomes, not the kind of
label.

The data-size cells intentionally measure the ordinary practical curve where
both unique examples and updates grow. They do not identify whether diversity
or repeated exposure caused a gain. Fixed-exposure configurations
(2,000×9, 6,000×3, 18,000×1) exist, but are gated on this screen to avoid an
unnecessary broad sweep. Later architecture comparisons will use the chosen
data/K point and information-matched flat controls.

The trajectory itself remains privileged: corruptions are made from the gold
solution's token pool and training follows the exact inverse corruption stack.
The terminal clean buffer is a prediction target. These facts prevent claims
of natural editing data or non-oracle planning.

## What happened

No valid trained-model comparison exists yet. The previous five jobs all
failed before their first optimizer step. The corrected data audit produced:

| Check | Observed result | Interpretation |
|---|---:|---|
| exact terminal recovery | 128/128 | every synthetic repair path reverses its corruption |
| multi-step collapse | 0/128 | official nested steps survive |
| official solution tokens | median 166.5; p95 251.95 | buffers are long, but steps are bounded |
| official steps | median 9; max 17 | the encoder sees a real step sequence |
| maximum step length | p95 43; max 62 | well below the 320-token safety cap |
| repairs per trajectory | mean 11.14; range 6–16 | planned horizons cover meaningful drift |
| path/minimum-edit ratio | mean 1.029; max 1.25 | paths are close to minimal, not padded arbitrarily |
| exact CF outcomes at K=4 | 5,704 across 1,426 states | four alternatives were available at every visited state |

Focused unit tests pass, all launch configurations compose, and a tiny CPU
K=2 optimization step had finite nonzero factual and counterfactual losses.
These are implementation gates only.

## The intuitive picture

![Diagram showing a clean multi-step solution, synthetic corruptions, an exact inverse expert path, and several mechanically executed alternatives from the same observed state.](decision.svg)

The figure separates two axes that are easy to confuse: more independently
corrupted solutions increase state diversity, while larger K increases local
action coverage at each state. The pilot varies one axis at a time around a
common 2,000-trajectory anchor.

## The technical details

The buffer is a nested list of official solution steps. Literal edits use a
flattened token position, mapped back to the retained step and offset. The
online buffer encoder feeds a causal action-conditioned latent predictor; an
exponential-moving-average encoder supplies next-buffer targets. For each
alternative action, its exact copied outcome buffer is independently encoded
by the same target encoder. The counterfactual loss is the existing normalized
latent outcome-prediction objective; no preference loss is active.

Primary diagnostics are held-out factual one-step LN-L1, exact-counterfactual
one-step LN-L1, recursive LN-L1 at horizons 1/2/4/8, and the shuffled-action
causal falsifier. Persistence is the no-change baseline. Errors are stratified
by edit operation and trajectory depth. State feature standard deviation and
effective rank gate collapse. Terminal-goal geometry is reported only as a
privileged diagnostic and cannot establish deployment-time planning.

The K screen retains the smallest K within 2% of the best candidate only if it
improves held-out counterfactual or recursive error by at least 5% relative to
K=0, while factual error and effective rank worsen by no more than 2%. If none
passes, counterfactual outcome prediction is removed. One seed selects cells;
seeds 1 and 2 are reserved for a chosen conclusion. Raw outputs will live
under `research/sequence_edit/logs/<run-name>/` with run summary, metrics,
stdout, and stderr artifacts.

The K=8 saturation cell is implemented but not in this round: the controller
rejected the seven-cell design at 18.92 projected GPU-hours because the
sequence-edit project allows 16 per round. The admitted six cells reserve
13.92 GPU-hours without using unrealistic timeout caps. K=8 runs only if K=4
improves without saturation and a later round is scientifically justified.

## What we can conclude

The old hierarchy result is invalid rather than negative. The repaired adapter
faithfully preserves the official step structure and exact synthetic recovery
on the audited sample. Exact unlabeled alternative outcomes are implemented
without target-relative quality labels. The six-cell pilot is technically
ready and answers a concrete selection question.

## What we cannot conclude

We cannot yet say that more data helps, that any K helps, or that hierarchy is
useful. The data-size curve confounds unique diversity with update count until
the fixed-exposure follow-up. The entire training task is synthetic,
candidate-privileged oracle denoising. One seed cannot support a final effect
claim. Exact one-step alternatives do not validate recursively imagined
off-support editing, autonomous proposal, closed-loop correction, or planning.

## What happens next

Run the six cells in parallel within controller resource guards. Then choose
the earliest data saturation point and smallest passing K. Only then run the
K=8 endpoint, if K=4 has not saturated, followed by the fixed-exposure
diversity cross and counterfactual-weight check; add a learning-
rate cross-check only if counterfactual gradients exceed twice baseline. After
those gates, test flat versus H4/H8, dense rollout, and LDAD removal, followed
by short/standard/long repair strata and confirmation seeds.

## Words used in this report

- **Buffer:** The editable multi-step solution text at one moment.
- **Counterfactual:** The exactly computed outcome of a different edit from the same observed buffer.
- **Candidate-privileged:** Data construction uses information about the gold answer that a deployed editor would not naturally have.
- **K:** Number of alternative edits and exact outcomes attached to each visited state.
- **LN-L1:** Absolute prediction error after separately normalizing latent vectors.
- **Effective rank:** A measure of how many independent representation directions are being used.
- **Oracle denoising:** Training where a known clean target defines both the damage and its exact repair.

## Questions for you

- After this screen, should the first follow-up prioritize separating unique-data diversity from update count, or quickly test hierarchy at the selected point?
- Is the main long-term goal a scientifically clean synthetic dynamics result, or should we prioritize replacing oracle inverse repairs with naturally proposed edits?
