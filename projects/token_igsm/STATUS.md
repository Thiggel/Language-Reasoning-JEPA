# Token-level iGSM status

The project pivoted on 2026-07-30. Earlier fixed-span, semantic-boundary,
support-constrained, and top-down CEM results are preserved as legacy and do
not validate the revised architecture.

The revised causal/indexing contract, pinned Qwen3.5-0.8B collector, observed
and counterfactual learners, three distinct oracle evaluators, value replay,
grounded CEM hooks, and exact-cache hierarchical MPC interfaces are
implemented. The active suite has 99 hierarchical tests; the broader active
repository suite has 429 passing tests.

The first bounded Stage 0/1/2 run completed on Grünau from commit
`d694c7802929ae536062bcd5777bb3c58de56f08`. It uses 50 verified training
problems, eight evaluation roots per split/horizon, 32 frozen-LM candidates,
and exact candidate re-encoding. This is candidate-privileged, single-seed
mechanism evidence, not a task-accuracy result.

Observed so far:

- exact token-endpoint planning is viable on some cells, but gains are not
  monotonic in horizon;
- learned rollout and exact-endpoint selection diverge increasingly on many
  long-span cells;
- counterfactual and sparse multistep training lower latent losses but do not
  improve the `k=8` learned planner on ID in this pilot;
- Mahalanobis and Euclidean tie on the ID `k=8` selection metrics, while
  cosine variants are worse;
- counterfactual generation consumes 50.9% of collection FLOPs and 83.7% of
  collection wall time; exact re-encoding consumes 47.9% and 14.9%;
- sparse rollout replay requires microbatching and takes about twice the
  one-step replay wall time.

Stages inherit exact admitted checkpoints and dataset-bound validity records.
Collector, learner, replay, CEM, and MPC paths emit component FLOP fractions
and wall-time diagnostics. Scaling to 7B remains gated on the two oracle
mechanism tests.

Sentence hierarchy remains gated. The current single-seed result does not
admit the sentence-waypoint worker. The corrected Alex seed is queued for
replication. Its operational state is
strictly `z1=E0_to_1(z0)`; there is no direct sentence projector. Macro-actions
and value learning remain gated on the nested sentence-waypoint worker.

Active specification: [`REVISED_PLAN.md`](REVISED_PLAN.md).
Current plot:
[`planning_effort_accuracy_readable.png`](../../runs/autonomy/token_igsm/2026-07-30-nested-language-first-five-paired-recovery-v3/nested-qwen08-first-five-gruenau-s0-recovery-v3/figures/planning_effort_accuracy_readable.png).
