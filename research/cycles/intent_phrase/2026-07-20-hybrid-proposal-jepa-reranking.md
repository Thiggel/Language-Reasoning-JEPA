# Cycle: calibrated proposal and JEPA reranking

Status: completed; proposal-only endpoint selected, validity gate failed

## Decision

Determine whether JEPA consequence simulation provides useful closed-loop
ordering after the planner retains the learned proposal evidence that created
its top-four candidate bank.

## Falsifiable comparison

Use the frozen aligned-history learning-rate-3e-3 checkpoint and identical
learned-catalogue candidates. Compare pure JEPA energy, pure proposal-prior
selection, and standardized hybrid scores with proposal weights 0.25, 0.5, 1,
2, and 4. Evaluate depths one, two, and four at exact lengths six and nine,
strict and plus-two budgets, over three fixed evaluation seeds.

The hybrid score standardizes JEPA energy and cumulative learned proposal cost
within each candidate bank before addition. This removes arbitrary unit-scale
differences without using labels, symbolic feasibility, or future menus. The
proposal cost contains only the behavioral prior and history-conditioned
availability estimates used to construct that same bank.

## Result patterns

- An interior weight that beats both endpoints and improves with depth retains
  hybrid JEPA planning for confirmation.
- An optimum at the proposal-only endpoint means current JEPA value/dynamics
  add no useful deployable ranking signal.
- An optimum at pure JEPA would contradict the previous 60-episode endpoint
  and require a seed/metric audit before interpretation.
- Invalid rates above .25 for every cell redirect proposal work to a
  length-balanced token prerequisite parser or policy language model.

## Primary metrics and gates

Primary: length-nine plus-two success and invalid-action rate. Secondary:
length-nine strict success, length-six success, and the depth response. All
cells share checkpoint, generated problem indices within seed, top-four roots,
beam width four, support weight ten, budgets, and simulator. No failed run or
cell is silently excluded. A viable hybrid must improve average plus-two
success over both endpoints without increasing mean invalid rate and must show
the same direction in at least two of three evaluation seeds.

## Implementation evidence

The planner now accepts a non-negative hybrid weight only for learned-catalogue
JEPA scoring and rejects prior-only mixing. Unit tests cover exact weight-zero
behavior, scale invariance, proposal-dominant ordering, and invalid protocol
combinations. The evaluator records the hybrid weight in every artifact.

## Result

All three evaluation jobs completed with all 63 declared artifacts. Pure JEPA
obtained .011 mean length-nine plus-two success. The best interior hybrid cell
obtained .278, while the proposal-only depth-one endpoint obtained .350 with
.506 strict invalid-action rate. No interior mixture beat proposal-only on the
primary metric, and no cell reached the .25 invalid-rate gate. Depth four
improved some weak mixtures but did not reverse the endpoint ordering.

The predeclared decision is therefore to stop mixture-weight tuning. The next
test preserves individual candidate/history tokens to determine whether pooled
phrase embeddings erased prerequisite-name matching.
