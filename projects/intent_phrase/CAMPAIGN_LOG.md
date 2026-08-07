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

## Results 2026-08-07 (one seed each unless noted; strict success D1..D16)

- Faithful TD-JEPA (arXiv:2510.00739): .133/.130/.117/.107/.107 — declines
  with depth. Takai GoalHead (JSAI 2026): .137/.037/.057/.060/.060 —
  collapses after D1. Same family signature as TD-Q/expectile: absolute
  calibrations don't survive recursive imagination. Details in the
  competitor-baselines report.
- Frozen-backbone Energy head (backbone frozen, head reset + retrained):
  recipe-shaped backbone .137/.433/.830/.840/.837; ranking-free TD-shaped
  backbone (init = baseline_td_q, value_detach=false so TD gradients DID
  shape it) .153/.360/.907/.927/.913 — ABOVE the jointly trained recipe at
  D4+ (single seed). Registered prediction (pure backbone clearly below)
  falsified: post-hoc distillation from a backbone that never saw ranking
  gradients works at full strength. Plain-JEPA backbone (no value/TD/rank
  gradients at all) now training to complete the triangle.
- Counterfactual K sweep (seed 0): K=0 .137/.237/.697/.877/.887 (shallow
  depths suffer, deep recovers); K=1 .183/.410/.870/.893/.893; K=8
  .157/.313/.783/.867/.857; K=32 .167/.313/.787/.880/.897. K=1..2 best at
  D2; odd K=8/32 dip needs seeds before claiming non-monotonicity.
- Theory probes (audit_theory_predictions.py, mix4 s0): T2 monotone
  invariance exact (planner metrics bit-identical under exp/cube/affine).
  T1 Kendall tau geometry-vs-steps-to-go only .46 mean — raw geometry is
  moderately ordinal; ranking head adds real information. T4: best-of-R
  label agrees with exact optimal root ordering .48 at H=1 but .83+ at
  H>=2 — quantifies why depth-1 is weak and lookahead scales. Registered
  in the theory report before the pure-backbone cell returns.
- Negative-results appendix compiled (17-row table + LaTeX):
  `research/reports/intent_phrase/2026-08-07-negative-results-appendix/`.
- Repo: full test suite green again (703 passed; consolidation breakage in
  token_igsm packed attention + samplers restored).

## Running / queued (2026-08-07)

- Counterfactual K in {3, 16, 64} still training (round
  `2026-08-07-intent-counterfactual-scaling-v1`, snapshot `9ca28ee`).
- Full-catalogue recipe TRAINING cell `mix4-fullcat-s0-v1` (gruenau10 GPU0,
  snapshot `b788372`): ranking alternatives from the full catalogue,
  inherited eval also plans full-catalogue at every depth (no feasibility
  oracle anywhere).
- Plain-backbone chain (gruenau1 RTX 6000): `plain-backbone-s0-v1` trains
  backbone objectives only (no value/TD/ranking gradients), then
  `frozen-backbone-plain-s0-v1` distills the Energy head on it frozen.
- Interface controls (owner's directive 2026-08-07): full-catalogue evals of
  recipe s0 + faithful baselines (no feasibility oracle, no symbolic future
  menus, invalid=noop) running as
  `2026-08-07-intent-interface-controls-v1/eval-fullcat-noop-s0-v1` on
  gruenau10 GPU1. Next: full-catalogue TRAINING cell; then learned action
  prior p(u(a)|s) (Gaussian MLP over action embeddings, port of
  sentence-vjepa VariationalAction) so planning needs no menu at all, and
  LM baselines move to free generation (parse-or-invalid). Report every
  method under both protocols.
- `run_intent_terminal_energy_eval.sh` now does ONE generous run per depth
  (MAX_SLACK=4, `slack_curve=true`): exact success at every slack 0..4 +
  per-episode excess steps (`slack_curves` in metrics.json; legacy per-slack
  rows kept for collectors). ~2x cheaper eval; use everywhere.

## Next steps (priority order)

1. Seed replication for paper tables: faithful baselines (td_jepa,
   goal_head), frozen-backbone (both inits), K sweep — 3-5 seeds each as
   GPUs free up. All evals via the slack-curve script.
2. Faithful iGSM + near/far length OOD as the primary generalization axis;
   looped-LM test-time-compute comparison with real FLOPs.
3. Theory T3 (margins vs compounded drift): needs light planner
   instrumentation to record root-candidate score gaps per depth; then the
   saturation-depth prediction test. T1/T2/T4 done (see results above).
4. Re-run five-seed headline evals with the slack-curve evaluator for full
   excess-step curves in the paper figures.
5. Stabilizer sweep (owner 2026-08-07, run AFTER menu-free recipe locked):
   backbone variants {EMA/stopgrad/online_nosg} x {VICReg/SIGReg/VISReg/
   LDAD/none} under the FROZEN distillation protocol (removes the EMA-label
   confound; labels from the frozen encoder). Stage 1: ~8 single-seed
   screening cells incl. Delta-JEPA claim (LDAD, no EMA, no stopgrad) and
   collusion control (online_nosg). Ports: SIGReg/VISReg/latent-LDAD from
   sibling tracks; state_target switch already in DiscourseJEPA.
6. Causal-predictor endpoint-Energy cell (needs training-time recursive
   imagination for the transformer predictor; guard currently blocks it).
7. Transfer to a second established environment (multi-hop logical
   reasoning) once the iGSM matrix is locked.
