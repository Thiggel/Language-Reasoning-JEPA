# Token-level iGSM status

The project pivoted on 2026-07-30. Earlier fixed-span, semantic-boundary,
support-constrained, and top-down CEM results are preserved as legacy and do
not validate the revised architecture.

The revised causal/indexing contract, pinned Qwen3.5-0.8B collector, observed and
counterfactual learners, three distinct oracle evaluators, value replay,
grounded CEM hooks, and exact-cache hierarchical MPC interfaces are
implemented and covered by CPU tests. The next scientific decision is Stage
0/1/2: validate a small verified iGSM manifest, measure frozen-LM proposal
coverage, then test flat token-space oracle endpoints with learned and exact
candidate encoding.

Stages inherit exact admitted checkpoints and dataset-bound validity records.
Collector, learner, replay, CEM, and MPC paths emit component FLOP fractions
and wall-time diagnostics. Scaling to 7B remains gated on the two oracle
mechanism tests.

Sentence hierarchy remains gated on that result. Its operational state is
strictly `z1=E0_to_1(z0)`; there is no direct sentence projector. Macro-actions
and value learning remain gated on the nested sentence-waypoint worker.

Active specification: [`REVISED_PLAN.md`](REVISED_PLAN.md).
