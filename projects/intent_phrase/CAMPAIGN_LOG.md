# Intent-phrase campaign log (living handoff)

_Keep short. Compress completed stages into a few lines; details live in
`research/reports/intent_phrase/<date>-*/REPORT.md` and are mirrored to
`/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/`. Last update: 2026-08-07._

## Frozen method (2026-08-07)

Horizon-blind endpoint Energy `E(root, imagined endpoint, z_0)` — no depth
input — logistic pairwise ranking of recursively imagined endpoints, labels =
EMA-latent distance of true rollout outcomes to the encoded solved state
(goal used only in training labels). Training horizons {1,2,4,8}, R=4
rollouts, K=2 alternatives, 0.25 root pair-difference auxiliary, no dense
loss. Cell variant `mix4_aux025_nohorizon`, snapshot `ee49091`.

**Five-seed headline (300 val episodes, root-balanced beam B=8, feasible menu
at D1, symbolic future menus [candidate-privileged] at D>1):**
strict .126/.450/.837/.879/.884 at depths 1/2/4/8/16
(slack-2 .541/.863/.979/.991/.995; random .053, first-feasible .247, oracle 1.0).

## Decided (details in dated reports)

- Horizon input to the Energy head: unnecessary (5 seeds). H16 training
  rollouts: harmful dilution. Fixed-H4: incoherent off-support (.29 at D2).
- Sparse prefix supervision, straightening, monotonicity (raw + projected):
  rejected (2026-08-06 factorial, all cells far below plain endpoint recipe).
- TD-Q (SARSA) baseline: .19 D1, no depth scaling. Expectile goal-value:
  near random. TD-as-shaping-auxiliary: no gain, D2 regression.
- Non-residual MLP prediction: roughly neutral (~1 SD below). Residual choice
  is second-order; predictor class (MLP vs causal transformer) is first-order.
- Repo: researchctl removed 2026-08-06; paper lives in
  `/vol/home-vol2/ml/laitenbf/TextJEPA-paper/` (own git).

## Running / queued (2026-08-07)

- Counterfactual-scaling sweep: frozen recipe with K in {0, 8, 32} on
  gruenau10 (round `2026-08-07-intent-counterfactual-scaling-v1`, snapshot
  `2d025b6`); K in {1, 3, 16, 64} queued for free GPUs. K=0 uses the new
  `data.geo_rank_factual_only` flag (same-root continuation ranking only;
  H=1 rows contribute no pairs by construction).
- Faithful-baseline implementation agent (worktree): exact TD-JEPA
  (successor features + task embedding, arXiv:2510.00739) and Takai GoalHead
  (JSAI 2026) score modes + cell variants.
- Single-pass slack-curve evaluator landed (`slack_curve=true` in plan.py):
  one generous-budget run yields success_by_slack for all N + per-episode
  excess steps. Use for all new evals.

## Next steps (priority order)

1. Single-pass evaluator: one generous-budget run records first-solution
   steps; derive full slack-N curve + overall accuracy (policy ignores
   budget, so exact). Use it for everything below.
2. Faithful baselines: TD-JEPA (arXiv:2510.00739, successor features + task
   embedding) and Takai et al. (JSAI 2026, GoalHead g(z_0) -> predicted goal
   latent + distance planning). Specs in
   `research/reports/intent_phrase/2026-08-06-competitor-energy-baselines/`.
3. Ablation matrix on the frozen recipe: counterfactual K in {0,2,3,8,16,32,64};
   frozen-backbone Energy head; per-component/objective ablations.
4. Faithful iGSM + near/far length OOD as the primary generalization axis;
   looped-LM test-time-compute comparison with real FLOPs.
5. Theory: identifiability propositions (ordinal geometry is the invariant;
   ranking is the canonical distillation) + pre-registered empirical tests.
6. Causal-predictor endpoint-Energy cell (needs training-time recursive
   imagination for the transformer predictor; guard currently blocks it).
7. Negative-results appendix table for the paper.
