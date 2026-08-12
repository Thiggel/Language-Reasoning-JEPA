# Intent-phrase campaign log (living handoff)

_Keep short. Compress completed stages into a few lines; details live in
`research/reports/intent_phrase/<date>-*/REPORT.md` and are mirrored to
`/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/`. Last update: 2026-08-12._

## RECIPE UPDATE (2026-08-08): LDAD added to the recipe (new headline)

Out of the owner's stabilizer sweep. New headline recipe =
`mix4_aux025_nohorizon` + `model.observed_action_ldad=true,
objective.observed_action_ldad.weight=1` (EMA+stopgrad+VICReg kept).
FIVE SEEDS (strict D1..D16): .202+-.049 / .729+-.026 / .885+-.026 /
.965+-.015 / .975+-.012 — dominates the 2026-08-07 recipe
(.126/.450/.837/.879/.884) at EVERY depth; D2 +.28 fixes the known
shallow-depth weakness (theory T4: best-of-R label noisiest at H=1;
LDAD's action-displacement decoding sharpens one-step geometry).
Frozen-protocol check PASSED (1 seed): head distilled from the FROZEN
LDAD backbone .183/.763/.887/.940/.940 — the gain is in the geometry.
Runner-up: SIGReg replacing VICReg (5 seeds .143/.434/.888/.959/.959,
frozen check .953 D16) — deep-depth gain only. SIGReg+LDAD without
VICReg does NOT compose (.843 D16); three-way combo running.

## Frozen method (2026-08-07, superseded above for the headline)

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
- Frozen-backbone triangle COMPLETE (1 seed each, identical frozen
  distillation protocol, strict D1..D16):
  plain JEPA-only backbone .100/.257/.643/.773/.783;
  TD-shaped backbone (td_q gradients, no ranking) .153/.360/.907/.927/.913;
  ranking-shaped backbone .137/.433/.830/.840/.837;
  joint-training reference (5 seeds) .126/.450/.837/.879/.884.
  Reading: (a) post-hoc distillation works on ALL three — ordinal
  structure largely emerges from JEPA objectives alone (plain .78 at D16
  vs random .05); (b) value-gradient shaping of the backbone adds ~.1
  (plain vs TD is the clean pair — same cheap data settings); (c) TD is a
  good geometry SHAPER but a bad SCORE (TD-JEPA/TD-Q planners fail with
  depth, yet TD-shaped backbone + ranking head is the best cell so far) —
  vindicates the owner's "TD as an ingredient" idea in modified form.
  Caveat: ranking-shaped backbone saw richer data (H mix, R=4) than
  plain/TD (H=1, R=1), so (b) uses plain-vs-TD only. Triangle COMPLETE at
  3 SEEDS (strict D16 mean+-sd): plain .786+-.015 < ranking-shaped
  .829+-.009 < TD-shaped .902+-.017 — all pairwise gaps non-overlapping;
  the ~.12 plain-vs-TD value-shaping gain at depth is the paper claim.
  Note: each run dir holds TWO eval sets — the paper-relevant pure
  endpoint-Energy planner is under `horizon_only/`; the root-level
  slackcurve files are a hybrid local-pruning variant (worse deep) —
  always collect from `horizon_only/`.
- Counterfactual K sweep COMPLETE (seed 0, D2 strict): K=0 .237, K=1 .410,
  K=2 .450 (5 seeds), K=3 .323, K=8 .313, K=16 .407, K=64 .407. Deep depths
  all ~.85-.93. Two caveats found: (a) K saturates at catalogue size
  (n_vars <= 12, alts = others[:K]) — K=16 and K=64 are bit-identical;
  report the axis as K in {0,1,2,3,8,all}. (b) K=0/8/32 ran from the
  pre-merge snapshot, K=1/3/16/64 post-merge; the merged baseline heads
  shift RNG at init, so same-seed cross-snapshot cells are NOT directly
  comparable — treat cross-snapshot differences as containing an extra
  seed-sized noise term. Robust conclusion: K=0 clearly hurts shallow
  depths; any K >= 1
  is equivalent within single-seed noise. Recipe K=2 stands; exact K is
  second-order. K=0 s1 replicates s0 (D2 .263 vs .237; deep ~.88) — the
  K=0 shallow penalty is now 2-seed. Full-curve seeds only if the paper
  wants error bars on every K.
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
- Interface controls RESULTS (2026-08-08): WITHOUT the feasible menu every
  method collapses to ~0 strict success at every depth (invalid-action
  rates 73-100%, invalid=noop): recipe s0 evaluated full-catalogue;
  TD-JEPA and GoalHead evaluated full-catalogue; recipe RETRAINED with
  full-catalogue ranking candidates; recipe + flat Gaussian action prior
  (learned_catalogue, top-8/top-3/top-2, with and without feasibility
  gate). Collapse is uniform — no method is privileged by the menu
  relative to another. Diagnosis: the prior has signal (correct action
  mean rank 3.6 of ~9 catalogue entries) but the PAIRWISE feasibility head
  is uninformative (58% acc, near-constant logits) because feasibility in
  iGSM is relational (all prerequisites executed). Fix committed
  (`1321619`): history-attention feasibility head wired into the intent
  track (causal executed-action history in training + planner gating);
  cell `mix4-prior-hist-s0-v1` came back collapsed for a DIFFERENT reason:
  the recipe experiment sets value_detach=false, so the prior NLL
  collapsed the action encoder (action_std 2.4 -> 0.002 in one epoch).
  Fixed (`d2f1cf5`: prior/feasibility supervision always detaches inputs);
  rerun `mix4-prior-hist-s0-v2` had HEALTHY embeddings (action_std 7.5)
  and STILL fails menu-free (~0 strict, invalid 72-82%, feasibility acc
  52% = chance, prior rank ~4.3/9). CONCLUSION (3 training iterations):
  menu-free planning fails for every method because the pooled state does
  not expose per-candidate prerequisite structure (it lives in the prompt
  text; nothing in JEPA training forces it into the state). This is the
  paper's honest interface-axis story: headline WITH menu (fair — uniform
  collapse without it for all methods), menu-free as a documented
  limitation + the collapse control. LM free-generation baselines
  predictably consistent; skip unless a reviewer asks.
  CLOSED 2026-08-10 after two final escalations on the LDAD backbone:
  Gaussian-NLL prior (rank ~5, menu-free ~0) and CONTRASTIVE prior
  (catalogue-softmax CE trained on exactly the plan-time ranking; ends AT
  the chance floor 2.57, menu-free .000-.030). No further menu-free work
  via STATE-readout heads; see interface-controls report.
  CORRECTED 2026-08-11 (state-readout-controls report): the information IS
  in the state — oracle probes decode feasibility from (s_t,u) at AUC .935
  (resolvedness .982), and the same head trained offline lands 0.47 nats
  BELOW chance. The failure is the next-action-imitation objective (rewards
  preference, barely rewards the conjunctive parent-lookup), not missing
  information. Say "imitation-trained state readouts stay near chance;
  cycle-consistency recovers the same information through the trained
  dynamics" — NOT "the state lacks the information".
  REOPENED 2026-08-10 via the DYNAMICS: LDAD cycle-consistency — score
  each catalogue candidate by how well the LDAD displacement decoder
  reconstructs the candidate's own phrase from the predictor's imagined
  displacement (feasible/infeasible AUC .94, no oracle, no training),
  mask the planner's own executed actions. First real menu-free numbers
  (`2026-08-10-intent-ldad-cycle-v1`, LDAD checkpoint, top-2): slack-4
  .25-.31 at ALL depths, invalid 24-43% (all state-readout attempts: ~0).
  Top-3 eval running. Owner position 2026-08-10: catalogue enumeration
  (variable names, prompt-derivable) is acceptable for the paper — the
  privileged part was feasibility, now learned. OPTIONAL STRETCH: fully
  open-ended proposals via a state-conditioned intent-phrase generator
  head vetted by cycle-consistency; would let LM baselines run pure
  free-generation (parse-or-invalid) with no candidate naming anywhere.
- STABILIZER SWEEP UNBLOCKED (owner gated it on "the eventual menu-free
  recipe"; that recipe is now a documented negative result, so the sweep
  runs on the standard with-menu frozen recipe). Stage 1 screening
  launched on gruenau1 (round `2026-08-08-intent-stabilizer-sweep-v1`,
  snapshot `6cf9424`, joint-training screening — frozen-protocol
  replication for survivors): stab-online-nosg (collusion control, no
  EMA/stopgrad targets), stab-ldad-only (Delta-JEPA claim: LDAD, no
  EMA/stopgrad, no VICReg), stab-sigreg (SIGReg replaces VICReg).
  Reference rows already exist (recipe 5 seeds = EMA+stopgrad+VICReg).
  SIGReg objective existed in-tree; VISReg port still pending (stage 2).
  LDAD + full recipe (EMA+stopgrad+VICReg+LDAD, 1 seed):
  .213/.757/.867/.943/.960 — a LARGE shallow-depth gain (D2 .757 vs
  recipe .450, D1 .213 vs .126), deep on par with SIGReg. Matches theory
  T4 (best-of-R label weakest at H=1): LDAD's action-displacement
  decoding plausibly sharpens one-step geometry. REPLICATED at 3 seeds:
  D2 .741+-.021 (vs recipe .450+-.031), D16 .972+-.012 (vs SIGReg
  .959+-.011) — LDAD+recipe is the best variant overall so far.
  SIGReg+LDAD (i.e. LDAD with VICReg REPLACED by SIGReg, 1 seed): does
  NOT compose — .130/.403/.783/.840/.843, worse than either alone.
  VICReg seems necessary for the LDAD gain. Candidate headline:
  LDAD+EMA+stopgrad+VICReg; seeds 3-4 + frozen-protocol distill from the
  LDAD backbone running on gruenau1.
  STAGE-1 RESULTS (strict D1..D16, 1 seed): online_nosg+VICReg
  .093/.317/.670/.727/.727 and LDAD-only .080/.223/.637/.723/.720 — both
  train WITHOUT collapse (Delta-JEPA's no-EMA/no-stopgrad claim holds for
  stability) but lose ~.16 deep vs the reference .884: EMA+stopgrad buys
  real performance here, not just anti-collapse. SIGReg (replacing
  VICReg, EMA+stopgrad kept): 3 SEEDS .877/.917/.873 at D4 and D16
  .950/.970/.950 — D16 .957+-.012 vs reference .884+-.023: a REPLICATED
  ~.07 deep-depth improvement over the frozen recipe. Seeds 3-4 running
  to reach the 5-seed headline convention; if they hold, the paper
  recipe becomes SIGReg (candidate `mix4_aux025_nohorizon` +
  objective.sigreg.weight=1, vicreg 0) and downstream tables re-anchor.
- Learned action prior IMPLEMENTED (commit bb300f5): GaussianActionPrior
  head (state -> Gaussian over action embeddings, NLL on observed actions),
  planner `candidate_interface=learned_catalogue` (prior-filtered catalogue
  top-k/top-p, no menu, no feasibility oracle), `mean_prior_rank`
  diagnostic; suite 731 green. First menu-free training cell
  `mix4-prior-s0-v1` (variant `mix4_aux025_nohorizon_prior`, snapshot
  `f27403a`) RUNNING on gruenau1 GPU0: recipe + full-catalogue ranking
  candidates + prior + support head, eval fully menu-free (top-k 8).
- Five-seed headline slack-curve re-eval COMPLETE
  (`2026-08-07-intent-headline-slackcurve-v1/aggregate_slack_curves.json`):
  strict means reproduce the frozen headline exactly
  (.126/.450/.837/.879/.884); mean excess steps when solved 2.04/0.96/
  0.24/0.17/0.15 at D1..D16. Figure-ready.
- Frozen-backbone triangle now 2-SEED on all three legs (s1: plain
  .107/.250/.643/.797/.803; TD-shaped .143/.383/.893/.917/.910;
  ranking-shaped .133/.390/.807/.830/.830 — all replicate s0).
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

## 2026-08-11: open-ended proposers built; domains unlocked; length-OOD running

- Owner directives: (a) implement all four contract domains (faithful iGSM,
  ProofWriter, PlanBench Blocksworld, text ALFWorld) via supervised
  subagents; (b) screen MULTIPLE fully open-ended proposal mechanisms
  (generator head AND CEM-in-embedding-space with decoder grounding +
  off-manifold control); (c) launch every paper-ready experiment in
  parallel. Confirmed: nothing had been trained on faithful iGSM yet.
- Open-ended proposers implemented + merged on main (5f89e47..f0fae8c):
  generator_cycle (state-conditioned AR phrase head, detached, sampled
  proposals vetted by LDAD cycle) and cem_cycle (CEM over action embeddings,
  training-embedding Gaussian prior pinned pre-eval-override, scale-invariant
  off-manifold penalty, LDAD decode for grounding). One shared decode/score
  module (planning/ldad_decode.py), one phrase parser (igsm/render.py),
  unified proposal diagnostics (proposal_recall, proposal_parse_rate —
  NOTE: parse-rate denominator differs generator=all samples vs CEM=elites).
  803 tests green. Adversarial review agent with failing-test mandate
  running over the full diff before any GPU spend.
- Observed-action planner merged: plan.py can now run slack-curve evals for
  ProofWriter/Blocksworld/ALFWorld through the same endpoint-Energy code
  path as stylized iGSM (evaluate_observed_action_planning; oracle row =
  privileged expert replay, labeled). Faithful iGSM evaluator gained
  single-pass slack_curve. Slack-curve outputs now get _slackcurve suffix.
- ProofWriter real-scale data compiled: data/intent_phrase/proofwriter/
  19884/1962/2000 episodes, depths 1/2/3/5 (OWA, ProofWriter's own splits),
  validation clean (replay/goal/catalogue-recall 1.0, disjoint splits),
  smoke-trains. MANIFEST.json has provenance. Blocksworld needs more
  instance sets + split discipline; ALFWorld needs large collection run.
- Length-OOD (runs/autonomy/intent_phrase/2026-08-11-intent-length-ood-v3):
  LDAD seeds s0-s4, feasible_menu lookahead 16. Train lengths 3-9; bands
  near 10-12 (nvars 20-28), far 13-16 (32-48), extreme 17-20 (40-64),
  strict length sampling. Two earlier launches were WRONG (EVAL_DEPTHS is
  lookahead, not problem length; default graphs can't reach length 10+) —
  discard 2026-08-11-...-v1. First result s1 near: mean_necessary 10.65,
  strict .047, slack-4 .233 (vs ~.96 strict in-dist) — big length drop.
  Graph-size confound control queued: length 3-9 at nvars 20-28
  (ctrl-nvars cells, auto-start when band evals finish).
- Ops notes: this session runs ON gruenau1 — pkill -f patterns match our own
  monitors/shell (use PID kills or bracket-escaped patterns). Parallel
  "predictive-state" subproject commits on main are NOT intent_phrase; leave
  alone.

## 2026-08-11 (cont.): probe verdict; codebook stage-1 negative; decoder stop-token fix

- STATE-READOUT CONTROLS (research/reports/intent_phrase/
  2026-08-11-state-readout-controls/): (A) "info absent from state" REFUTED
  — feasibility probe AUC .935, resolvedness .982 (oracle labels,
  candidate-privileged diagnostic). Residual is the OBJECTIVE: four
  imitation-trained heads (71k..1.65M params, incl. cross-attention over the
  sentence sequence) all stall >=0.48 nats above the feasible-entropy floor
  with feas. AUC <=.83, while a directly supervised same-shape MLP hits
  .935. One-hot ablation proves the conjunctive parent-lookup is the hard
  part (.935 -> .755). Campaign-log + interface-report overclaims corrected.
- CODEBOOK (owner idea, stage 1 eval-time k-means over training action
  embeddings, candidate_interface=codebook_cycle, committed da421db):
  NEGATIVE on existing LDAD checkpoints. Two causes found: (1) LDAD decoder
  free-decoding never stops (PAD position unsupervised — same defect family
  the adversarial review fixed in the generator loss); mitigated eval-time
  by terminator ('.') truncation in greedy_phrases. (2) After the fix,
  grounding is the blocker: codes decode to plausible but WRONG-problem
  variable names (1/64 parseable at a real state; own-catalogue embeddings
  reconstruct exactly teacher-forced but only 2/9 free-decoded). Free
  decoding from off-catalogue embeddings is not grounded by state
  conditioning on current checkpoints. Stage 2 options if pursued: supervise
  LDAD PAD position (retrain), or rely on the generator head (trained for
  free generation, PAD fix in). Screening evals eval-codebook-k{64,128}
  recorded the honest 0-parse numbers pre-fix.
- Length-OOD first number (s1, near band 10-12 steps, nvars 20-28):
  strict .047, slack-4 .233 (in-dist ~.96 strict) — large drop; graph-size
  control cells queued.

## 2026-08-11 (cont. 2): frozen-state sentence decoder (owner-requested renderer)

- Report: research/reports/intent_phrase/2026-08-11-state-decoder/ (mirrored).
  JEPA-pure readout (frozen backbone, zero-grad asserted). TRUE states:
  77% exact sentence / 96% token / 96% VALUE accuracy; final-answer head 94%
  (chance 4%) — states are a faithful decodable record incl. arithmetic.
  IMAGINED states: form survives (token .92 at d1-2), values collapse
  (.32 -> chance by d5). Controls: prompt-only floor .248 value (one
  predictor step buys +7pts); one-step-from-true-prefix NO better than
  compounded rollout at matched depth => SINGLE-STEP fidelity, not
  compounded drift, is the bottleneck; predictor contributes structure
  (token .561 -> .923), not computation. Paper story: predictor propagates
  plan-relevant geometry (what cycle-consistency/Energy read), not
  arithmetic; explains why calibrated-value baselines fail with depth.
  Feeds theory T3 (drift is immediate, not accumulated).

## 2026-08-11 (cont. 3): open-ended screen — generator negative, CEM eval launched

- gen-ldad-s0-v1 (LDAD recipe + state-conditioned action-generator head,
  PAD fix in, snapshot 318d6c2) trained clean (LDAD token acc .999,
  generator CE plateau 0.62/token) but generator_cycle planning fails:
  parse rate .132, no-usable-proposal in 98.7% of episodes, strict .013
  (random-with-menu .050). Direct sampling from the checkpoint shows the
  SAME grounding failure as the codebook: phrases are fluent, correctly
  formatted, but recombine adjectives/nouns across problems ("square
  beads", "large apples" for a problem that has neither) and are all leaf
  lookups; lowering temperature only shrinks the sample set, parseable
  fraction stays ~1-2 of 16. Consistent conclusion across codebook +
  generator: emitting THIS problem's variable names from the pooled state
  is the shared bottleneck for every catalogue-free proposer — same
  conjunctive-binding gap the state-readout probes localized. ldad_cycle
  (catalogue enumeration + cycle feasibility) remains the menu-free result.
- cem_cycle eval launched eval-cem-s0-v1 (frozen stab-ldad-ema-s0, fresh
  snapshot 734bc58 WITH the greedy_phrases terminator fix; pop 64, elites 8,
  3 iters, prior anchor 0.1). Note: first launch used a pre-fix snapshot
  and was killed before producing output. Expectation: same grounding
  ceiling (CEM decodes phrases from optimized embeddings), screen closes
  the "catalogue-free" question either way.
- cem_cycle result: parse rate .032, zero usable proposals in 100% of
  episodes, strict .000 — same grounding failure, worst of the three.
  SCREEN CLOSED: all catalogue-free proposers (generator/CEM/codebook) fail
  on emitting this-problem variable names; ldad_cycle (catalogue + cycle
  feasibility) is the paper's menu-free result. Report:
  research/reports/intent_phrase/2026-08-11-openended-screen/ (mirrored).

## 2026-08-11 (cont. 4): faithful-iGSM admission gate — one real defect found+fixed

- Ran the paper admission gate (run_igsm_paper_admission_gate.sh, tiny cells,
  gate-s0-v1). Passed: oracle 1.0 / random .11 strict bounds sensible,
  geometry beats random, dropout zero, EMA eval-mode invariants OK.
  DEFECT: the shuffled-actions control was bit-identical to the aligned cell
  — the faithful adapter (src/textjepa/data/faithful.py) silently dropped
  data.shuffle_actions via **_ kwargs. Fixed (80cb55e): alignment permuted
  last after all other randomness, stylized-dataset contract; regression
  test added (shuffle must change alignment, nothing else). Shuffled gate
  cell rerunning from the fixed snapshot; LR screen admitted once
  shuffle-hurts is confirmed.
- Shuffle-flag bug had a SECOND half: build_dataset's igsm_real branch
  enumerates kwargs and omitted shuffle_actions (fixed ce2581c, end-to-end
  verified). Tiny gate cell (64 problems) now differs but is ambiguous;
  scaled pair (512 problems, 200 eval episodes) running to settle
  shuffle-hurts. AUDIT NOTE for later faithful work: the faithful adapter
  also silently drops (via **_) stylized-only flags incl.
  invalid_action_mode, steps_range/strict_steps_range,
  geo_rank_factual_only, geo_rank_candidate_interface — any faithful
  length-OOD or invalid-mode experiment must add explicit support first,
  not assume the config key works.
- Scaled shuffle pair (512 problems): BOTH arms at the random baseline —
  the tiny gate model does not learn planning at that scale, so the
  shuffle-hurts criterion cannot be shown in tiny cells. Resolution: test
  the falsifier at real recipe scale (contract requires it as a causal
  falsifier anyway). LAUNCHED faithful-iGSM compact LR screen
  (2026-08-11-intent-faithful-screen-v1): LDAD headline recipe on
  data=igsm_real, LRs {1e-4,3e-4,1e-3,3e-3} seed 0, PLUS
  shuffled-actions arm at 3e-4 (real-scale gate check). Sequential chain
  on gruenau1 GPU 0, order 3e-4 -> 1e-3 -> shuffle -> 1e-4 -> 3e-3;
  snapshot ce2581c (both shuffle fixes in). All other gate criteria
  passed (oracle/random bounds, dropout, EMA invariants, geometry>random
  in the overfit cell).
- Screen launch #1 failed at hydra parse: the shared cell script passes
  stylized-only data keys. Ported to the faithful adapter (42fd571):
  geo_rank_horizons multi-horizon sampling (stylized-identical RNG stream)
  + geo_rank_rollout_for_h1 implemented; candidate_interface/factual_only/
  feasible_k/invalid_k/invalid_action_mode accepted at defaults, raise
  NotImplementedError otherwise (no more silent drops). CPU dry-run of the
  full override set trains. Chain relaunched from snapshot 42fd571.

## 2026-08-11 (cont. 5): standup-figure baselines — LM under all three menu conditions

Owner wants the depth-scaling figure's baseline to be a real sentence/token
LM (not TD-JEPA/GoalHead) across all three interface conditions (feasible
menu, full-catalogue non-oracle, no-menu free generation), plus our current
LDAD headline checkpoint re-evaluated non-oracle for a fair 3-way comparison
figure per condition. Stylized igsm only (LM checkpoints `runs/lm_intent`,
`runs/sentlm_intent` are stylized; faithful full-catalogue LM path is out of
scope here, raises NotImplementedError).

Already on record before this round: token-LM/sentlm-LM feasible-menu
(slack0 strict): token .830, sentence .635 (`runs/lm_intent/
plan_slack0_lm_intent.json`, `runs/sentlm_intent/
plan_slack0_sentlm_intent_decoder.json`). No prior full-catalogue or
free-generation strict-success numbers existed for either LM baseline
despite the code paths existing (`plan_lm.py`/`plan_sentlm.py` always used
`env.feasible_actions()`; `eval_generative_lm_baseline.py` never had output
saved for these checkpoints).

Added `candidate_interface=full_catalogue` to `plan_lm.py`/`plan_sentlm.py`
(previously feasible-menu only): candidates become
`range(len(problem.vars))`, invalid picks go through
`env.step_or_invalid` (invalid=noop), `invalid_rate` added to output,
output filename gets a `_full_catalogue` suffix so it doesn't clobber the
existing feasible-menu json. Scoped to stylized igsm + score/target_kind
`intent` only (faithful + outcome-scoring raise). 703-suite-adjacent tests
(`test_fixed_budget_runner.py`, `test_oracle_cem_contract.py`) still pass;
smoke-tested both scripts at n_episodes=8 on both interfaces before the
real run.

Launched (round `2026-08-11-standup-baselines-v1`, from the working tree,
not a snapshot — eval-only, no training):
- `ours-fullcat-s0-v1` (gruenau, cuda:2): `stab-ldad-ema-s0-v1/model/
  best.pt` (the 5-seed headline checkpoint's seed 0), candidate_interface=
  full_catalogue, EVAL_DEPTHS="1 2 4 8 16", 300 episodes, beam width 8,
  via `run_intent_terminal_energy_eval.sh` (same script/protocol as the
  headline depth curve, so directly comparable).
- `lm-baselines/` (gruenau, cuda:0), sequential: token-LM and sentence-LM
  full_catalogue (slack0, 300 episodes) + token-LM and sentence-LM no-menu
  free generation (`eval_generative_lm_baseline.py`, 300 examples,
  width=8, `problem_solve_rate` is the strict-success field to read).

Note for whoever reads this next: the LM baselines have NO planning-depth
axis (greedy one-step-at-a-time policy, no imagined lookahead), so the
"3 figures" (one per menu condition) should show our depth curve plus a
FLAT reference line per LM baseline, not 3 comparable curves — don't force
a depth axis onto the LM numbers.
- Screen attempt #2 surfaced three more faithful-port defects (all fixed,
  deb1e5f): (1) CRITICAL - horizon-mode GAR requires ga_rollout_actions,
  which faithful rollouts never emitted, so geo_horizon_rank (the recipe's
  key ranking loss) silently skipped on igsm_real; the timed-out lr3e4
  cell trained with the loss at 0.0 throughout. Now emitted + alignment-
  tested; dry run shows the loss active. (2) FaithfulPlanner lookahead>1
  crashed on _sequences' absorbing None padding (action_text(None));
  lr1e3 trained fine and died in eval. (3) 5h/epoch at num_workers=2
  (faithful generation is CPU-bound): cell script now takes NUM_WORKERS;
  chain relaunched with 16 workers, all cells reset (earlier partial
  checkpoints trained WITHOUT the ranking loss - discarded).

## 2026-08-12: codebook deep diagnosis (owner question) — impossible in principle

- Report: research/reports/intent_phrase/2026-08-12-codebook-diagnosis/
  (mirrored). Phrase space is combinatorial: leaf lookups saturate (100%
  verbatim coverage), compute phrases NEVER repeat (0/1367 across 300 val
  problems; 0 problems with necessary actions covered) — so no codebook
  size can supply them. Twist: the LATENTS are nearly reusable (NN dist
  0.77 vs within-problem separation 10.7) — owner's intuition right at the
  vector level — but the environment executes sentences, and the retrieved
  vector's phrase names the wrong problem's variables. Action embeddings
  are coarse role/op codes; names live in the state, not the action code.
  Menu-free via prompt-derived catalogue + cycle feasibility stands as the
  right claim; retrieval memories are ruled out in principle.

## 2026-08-12: LM-baseline + scaling program stood up (PREPARED, nothing launched)

Audit of the contract's six learned LM rows: all six have a TRAINING path
already (`scripts/train_lm.py` with `train.target_kind=intent`,
`scripts/train_sentlm.py` with `model.latent_target`, and `model.recurrent`
for the three weight-shared counterparts). Planning eval under the
feasible-menu interface also existed (`plan_lm.py`, `plan_sentlm.py`), but it
did NOT produce the JEPA slack-curve metrics: it emitted a single fixed-slack
success and its own ad-hoc dict. Fixed (d52bae5): both now aggregate through
`textjepa.planning.evaluate.aggregate_episodes` with `slack_curve=true` and
record `solved_at` (exact for a greedy policy that stops at the goal), so
`success_by_slack` / `excess_steps` / `invalid_action_rate` /
`mean_necessary` match `scripts/plan.py` field-for-field. `invalid_rate` is
kept as an alias; curve runs get a `_slackcurve` filename suffix.
`configs/plan.yaml` gained `score` and `eval_loops`.

New `scripts/run_intent_lm_screen_cell.sh` is the LM analogue of
`run_intent_horizon_energy_cell.sh`: trains one row, then evaluates it under
the identical feasible-menu protocol, writing `metrics.json` with
`slack_curves`. It keys curves by RECURRENT LOOP COUNT (contract loops
{1,2,4,8,16}) and sets `planning_depth_axis: false`, so a loop count can
never be misread as a JEPA simulated-transition depth.

Trained LM checkpoints that already exist (usable NOW for the frozen-feature
state-readout probes — `scripts/export_intent_representations.py` already
supports `--kind token_lm|sentence_lm`): token LM d_model=288 and sentence LM
(+/- latent) d_model=256, seeds {1,2,3}, on BOTH stylized (`runs/lm_intent*`,
`runs/sentlm_intent*`, `runs/sentlm_latent_intent*`) and faithful
(`*_faithful*`) iGSM, all at lr 3e-4 (faithful token LM at 1e-3), 20 epochs,
train_size 100000. NOTE these are off-contract widths (288/256, not
128/256/512) and off-contract seeds (1..3, not 0..4), so they are probe/
diagnostic material, not headline rows. NO recurrent LM checkpoint exists.

PREPARED (not launched):
- `runs/autonomy/intent_phrase/2026-08-12-intent-lm-screen-v1/` — compact LR
  screen {1e-4,3e-4,1e-3,3e-3}, width 256, seed 0, stylized iGSM, for token
  LM and sentence LM (8 cells + `run_chain_gpu0.sh`). EPOCHS=10
  TRAIN_SIZE=30000 BATCH_SIZE=32, matching the JEPA screen cells.
- `runs/autonomy/intent_phrase/2026-08-12-intent-width-scaling-v1/` — JEPA
  LDAD recipe (`mix4_aux025_nohorizon` + observed_action_ldad) at widths
  {128, 512}, lr 3e-4, seed 0, stylized iGSM (2 cells +
  `run_chain_gpu1.sh`). Width key is `model.d_model`; verified to
  instantiate at 6.05M / 22.82M / 89.41M params for 128/256/512. Width 256
  is already covered by `stab-ldad-ema-s0-v1`.
Both rounds run from the immutable snapshot `runs/autonomy/_code/d52bae5...`.

Still unimplemented (documented, not fixed): the three recurrent LM rows have
no screen cells yet (the runner supports them via `token_lm_rec` /
`sentence_lm_rec` / `sentence_lm_latent_rec`, but no checkpoint has ever
been trained, so their loop axis is untested); full_catalogue LM baselines
still raise on faithful iGSM; the LM rows have no matched-FLOPs axis
(they have no lookahead to trade compute against).

## 2026-08-12: predictor-variant falsifiers (code fix + PREPARED cells)

The paper's causal-falsifier list needs two predictor variants of the LDAD
headline recipe on stylized iGSM: a causal-sequence predictor (transformer
world model over the whole state/action history) and a non-residual MLP
predictor (predicts the next latent directly instead of a delta).

Fix: horizon-mode GAR (the recipe's ranking loss, which imagines a rollout
by stepping the predictor over encoded action tokens) used to raise
"horizon GAR currently requires the matched MLP predictor". Root cause: it
called the predictor once per horizon step with a single (state, action)
pair. For the causal predictor that call is a length-one sequence — it
resets the position embedding and throws away the factual prefix
s_0..s_t / a_0..a_{t-1}, i.e. silently degrades it to a Markov MLP.
`DiscourseJEPA._horizon_prefix_endpoints` now rebuilds the factual prefix
per anchor step (grouping rollout rows by their anchor t) and uses the
predictor's own history-carrying `rollout`; masked post-terminal suffix
actions stay absorbing exactly as before. The MLP path keeps the literal
old recursion, pinned by
`test_horizon_gar_mlp_path_matches_explicit_markov_recursion` (compares
model output against the explicit `s <- F(s,a)` loop, so existing
checkpoints are unaffected). Two new causal tests check gradients flow and
that the endpoints equal the history-carrying rollout and differ from the
old history-free recursion. tests/test_model.py: 67 passed.
Rollout-exposed GAR (`geo_rank_rollout_depths`) still raises for the causal
predictor — untouched, unused by this recipe.

Smoke-tested both variants for 1 epoch on CPU (tiny sizes) through
`scripts/run_intent_horizon_energy_cell.sh mix4_aux025_nohorizon` with
observed_action_ldad: non-residual (`model.predictor_residual=false`) trains
and evaluates end to end; causal (`model.predictor_kind=causal`) trains with
`train/geo_horizon_rank` active (nonzero), i.e. the loss is no longer
skipped or blocked.

PREPARED (not launched):
- `runs/autonomy/intent_phrase/2026-08-12-intent-predictor-variants-v1/`
  `pred-causal-s0-v1` (model.predictor_kind=causal) and `pred-nonres-s0-v1`
  (model.predictor_residual=false), seed 0, lr 3e-4, EPOCHS=10
  TRAIN_SIZE=30000 BATCH_SIZE=32, stylized iGSM, N_EPISODES=300
  BEAM_WIDTH=8 EVAL_DEPTHS="1 2 4 8 16" MAX_SLACK=4, CUDA_VISIBLE_DEVICES=0,
  to be run on gruenau2 from an immutable snapshot under
  `runs/autonomy/_code/`.

## 2026-08-12 (cont.): LM state-readout probes — feasibility signal NOT JEPA-specific

- Report: research/reports/intent_phrase/2026-08-12-lm-state-readout/
  (mirrored). Token LM .971 / sentence LM .961 / sentLM+latent .947 vs JEPA
  .935 feasibility AUC (own action embeddings; matched 16-d candidate:
  sentence LMs land AT JEPA, token LM above). Resolvedness tied everywhere.
  Conjunctive parent-lookup collapse replicates on all families. LMs decay
  LESS with depth than the JEPA pooled state. CONSEQUENCE: drop any
  "JEPA states carry structure LMs lack" phrasing; representation story is
  about USE (cycle-consistency converts signal into planning; LMs have no
  analogous mechanism), presence is universal. Also honest JEPA weakness:
  deep-prerequisite feasibility decays in pooled state. Caveats: single
  seed, training-campaign confound, not the matched-width protocol.
- Also this session: codebook_ground interface committed (9f9f1b7), 300-ep
  eval running; predictor variants fixed+tested (fa63cc9: causal predictor
  was silently Markov inside horizon GAR — real bug), cells queued on
  gruenau2 waiter; LM LR screen (8 cells) + width scaling {128,512} chains
  launched on gruenau2 (metric-parity fixes d52bae5).

## 2026-08-12 (cont.): long-trace retrain (owner-ordered redesign of length OOD)

- Owner: train on 15-25-step traces (like official iGSM's longer op counts),
  eval bands 25-30 ... 45-50; cancel superseded runs. CANCELLED: v3 extreme
  reruns (s2/s3/s4), ctrl-nvars cells/waiters, LM OOD evals on 3-9-trained
  checkpoints. v3 near/far 5-seed numbers stay recorded (near strict
  .055±.062, slack-4 .282±.255; far ~0; extreme 0 on 2 seeds; protocol was
  feasible_menu + candidate-privileged future menus at lookahead 16).
- Design probing: stylized generator cannot produce >~37-step cones at
  default leaf_prob .35; leaf_prob=0.1 (held FIXED across train and eval so
  length is the only axis) makes all bands samplable fast: train [15,25]
  @ nvars [30,60]; eval [25,30]@[60,120], [30,35]@[90,160], [35,40]@
  [110,180], [40,45]@[130,200], [45,50]@[150,220]. Longest episodes ~205
  chunks -> model.max_chunks=256; batch 16.
- LAUNCHED 2026-08-12-intent-long-traces-v1 on gruenau1: jepa-ldad-long-s0
  (GPU2, LDAD recipe lr 3e-4), tok-lm-long-s0 (lr 3e-3) -> sent-lm-long-s0
  (lr 3e-4) chained on GPU1. Band evals to follow after training.
- codebook_ground 300-ep eval COMPLETED — results to be read/reported.

## 2026-08-12 (cont. 2): results batch + REVISED OVERALL PLAN (owner-confirmed)

Results landed:
- pred-causal-s0 (causal-sequence predictor, guard fixed): strict by depth
  1/2/4/8/16 = .327/.670/.850/.963/.970 — comparable to the MLP headline
  (five-seed .202/.729/.885/.965/.975), single seed. Causal predictor is
  NOT a failure mode; report as architecture variant. pred-nonres running.
- Width scaling (stylized 3-9, lr 3e-4 s0): w128 .933 / w256 .975 (headline)
  / w512 .990 strict at depth 16 — monotone width scaling.
- LM LR selection RESOLVED (2 seeds): token LM -> 3e-3, sentence LM -> 3e-4
  (mean success tie-breaks per contract). Long-trace cells already use these.
- codebook_ground 300-ep: slack-4 .26-.28, proposal recall 1.0 — matches
  ldad_cycle (.29); owner idea works, added to matrix below.

REVISED PLAN (stylized iGSM long-trace = primary mechanism environment):
- S1 screens (RUNNING): jepa-ldad-long / tok-lm-long / sent-lm-long at
  steps 15-25, leaf_prob 0.1, nvars 30-60; + JEPA LR cross-check when free.
- S2 mains: 5 seeds LDAD; TD-JEPA + GoalHead baselines; token/sentence/
  sentence+latent LMs AND looped (recurrent) counterparts (loops axis
  {1,2,4,8,16}, never depth curves). Slurm (Lise/Alex) packaging.
- S3 eval matrix: {ID 15-25, 25-30, 30-35, 35-40, 40-45, 45-50} x
  {feasible_menu, full_catalogue (no-oracle stress), ldad_cycle,
  codebook_ground} x depth/loops {1,2,4,8,16}. codebook_ground carried as a
  co-equal menu-free interface (evidence label: environment-side grounding).
- S4 ablations (contract list) incl. predictor variants (causal, non-res)
  on long traces; width {128,512} rerun at long traces for the main table.
- S5 DETACHED DECODER on everything: frozen-state sentence decoder
  (JEPA-pure readout, zero-grad, as 2026-08-11) trained per final main
  checkpoint (and per domain), so every reported system renders executed
  plans to text; report decode accuracy alongside planning numbers. LM
  rows already produce text natively — matched decoder capacity per the
  contract's frozen-feature decoder protocol.
- Domains (faithful screen running; ProofWriter/PlanBench/ALFWorld agent
  finishing gates) and probe battery on final five-seed checkpoints
  unchanged from before.

## 2026-08-12 (cont. 3): plan corrections (owner)

- S5 CORRECTED: detached decoder is NOT a per-checkpoint reporting layer —
  it is the execution engine of the fully autonomous no-menu mode: propose
  (cycle/codebook) -> decoder EMITS next-step text -> model re-encodes its
  own text -> repeat until claimed done -> check FINAL ANSWER only. No
  oracle executor inside the episode. One decoder per main checkpoint
  (JEPA-pure, frozen states). NEW BUILD ITEM: autonomous self-rollout eval
  mode (planner loop over own generated text + answer check). Preregistered
  expectation from decoder controls: structure survives, arithmetic is the
  weak link -> honest end-to-end number whatever it is. Completes the
  interface ladder: feasible menu -> no-oracle menu -> menu-free w/ oracle
  executor -> fully autonomous text generation.
- Interface axis clarified: ONE training run per seed serves all
  interfaces; codebook/prior are eval-time fits on the frozen checkpoint
  (never trained in). S3 stands as 6 bands x {feasible_menu,
  full_catalogue, ldad_cycle, codebook_ground, autonomous} where applicable.
- Causal predictor: on-par result (.970 vs .975 strict@16) -> ablation
  table entry (recipe not predictor-specific), no headline treatment.

## 2026-08-12 (cont. 4): S2 mains + ProofWriter screen SUBMITTED to Alex

Local Gruenau GPUs are saturated, so the paper-ready training campaigns were
packaged for Slurm. Alex only: every Lise job pends with
`AccountOutOfComputeTime` (account bem00089 budget exhausted, confirmed live),
so Lise is unusable for overflow until compute time is restored.

Alex deployment (own filesystem): immutable snapshots under
`/home/atuin/c107fa/c107fa12/TextJEPA-autonomy/_code/<sha>` pushed with
`scripts/cluster/sync_snapshot_to_alex.sh`; runs under
`.../TextJEPA-autonomy/runs/autonomy/intent_phrase/<round>/<cell>/` with the
same `job.sh`/`state`/`exit_code`/`stdout.log`/`stderr.log`/`environment.json`
layout as Gruenau, so `scripts/cluster/sync_back_round.sh <round>` drops
results straight into the local tree. Shared env
`/home/atuin/c107fa/c107fa12/.venv` (torch 2.9+cu128) verified by running the
LDAD cell AND both untested recurrent LM rows end to end on CPU there first.
Cells queue with `--partition=a40,a100,rtxpro6k --gres=gpu:1`: a single-a40
submission was estimated to start six days out, across three partitions the
first six cells started within minutes. Alex caps single-node jobs at 24 h
(`--time=23:55:00`, inner watchdog 84600 s) — the 48 h wish for the recurrent
rows is not grantable there; if a rec cell TIMEOUTs it needs a resume path.

- A. `2026-08-12-intent-long-mains-v1` LAUNCHED, 45 cells, jobs 3989961-3990010
  (snapshot 5213602). Long-trace stylized iGSM (steps 15-25, nvars 30-60,
  leaf_prob 0.1, strict, max_chunks 256 / token LM max_len 4096), EPOCHS=10
  TRAIN_SIZE=30000 BATCH_SIZE=16 N_EPISODES=300 MAX_SLACK=4, JEPA depths
  {1,2,4,8,16}. Rows x seeds 0-4: LDAD recipe (lr 3e-4, one knob `LDAD_LR`
  pending the local 3e-4 vs 1e-3 cross-check), TD-JEPA and GoalHead baselines
  (lr 3e-4, as in the 2026-08-07 baseline round), token LM 3e-3, sentence LM
  3e-4, sentence+latent 3e-4, and the three RECURRENT LM rows (loop axis) with
  seed 0 as smoke and seeds 1-4 held on `--dependency=afterok`. No ablations.
- B. `2026-08-12-intent-proofwriter-lr-screen-v1` LAUNCHED, 12 cells, jobs
  3990030-3990041 (snapshot a2f035e); compiled ProofWriter data rsynced to
  Alex. LDAD + token LM + sentence LM at lr {1e-4,3e-4,1e-3,3e-3}, seed 0,
  EPOCHS=30. ENTRY-POINT FINDING (checked on CPU first, as ordered): the
  stylized-iGSM cell `run_intent_horizon_energy_cell.sh` CANNOT run
  ProofWriter — it hard-codes on-the-fly iGSM sampling keys absent from
  `configs/data/proofwriter.yaml` (`Key 'geo_rank_rollouts' is not in struct`,
  aborts before step 1) and its evaluator samples iGSM problems. Fixed by
  using the compiled observed-action path: added a `proofwriter` case to
  `run_compiled_domain_ldad_cell.sh` and wrote `run_compiled_domain_lm_cell.sh`
  (train_lm/train_sentlm with `data=proofwriter train.target_kind=intent`, then
  `eval_observed_action.py`, same information-matched interface as the JEPA
  row). All three cells smoke-tested end to end; 31 related tests pass.
  CAVEAT: compiled-domain evals report `metrics_by_excess_actions`, not the
  iGSM `slack_curves` shape; recurrent / sentence+latent LM rows are not wired
  for compiled domains.
- C. `2026-08-12-intent-faithful-mains-v1` PREPARED ONLY on Alex (5 seeds,
  `state=PREPARED`, lr placeholder `PENDING_LOCAL_SCREEN`); launch with
  `FAITHFUL_LR=<lr> SUBMIT=1 bash .../alex_submit_intent_faithful_mains.sh`
  once the local faithful LR screen decides.

New helpers (committed): `scripts/cluster/{alex_env.sh,sync_snapshot_to_alex.sh,
alex_submit_intent_long_mains.sh,alex_submit_proofwriter_lr_screen.sh,
alex_submit_intent_faithful_mains.sh,sync_back_round.sh,
write_cell_environment.sh}`. Per-round READMEs with the exact rsync-back
command live in each local round directory.

## 2026-08-12 (cont. 5): PlanBench Blocksworld + text-ALFWorld corpora built, admission gates run

Report: `research/reports/intent_phrase/2026-08-12-planbench-alfworld-unlock/`.
Round dir: `runs/autonomy/intent_phrase/2026-08-12-intent-domains-v1/`.

CORPORA (both frozen, with provenance MANIFESTs; `data/` is gitignored):
- PlanBench Blocksworld `data/intent_phrase/planbench_blocksworld/`: 3000 train
  / 60 val / 80 test episodes + a 48-episode 6-block length-OOD test set.
  Instance sets `generated_basic_3`, `generated_basic`, `generated` (official
  pool 263). Identity = (initial, goal) atom set minimised over all block
  RENAMINGS, so relabelled duplicates cannot straddle splits; val/test are
  official instances only. Rejections recorded (2420 held-out-identity, 505
  renaming-duplicate, 1683 literal-duplicate, 335 block-count).
- text-ALFWorld `data/intent_phrase/alfworld/paper_v1/`: 840 train / 80 val /
  80 test episodes, 12204/1111/1244 transitions, 48816/4444/4976
  counterfactuals (half deliberately infeasible). Splits are the official
  train / valid_seen / valid_unseen game sets, zero gamefile overlap, all six
  task families present in every split. 100% of counterfactual branches carry
  `teacher_rollout_actions`. Collected by 48+8+8 deterministic shard stripes;
  40 of 48 train stripes completed (2 wedged on games that outlive the 900 s
  episode timeout, 6 were still slow-collecting when the round was frozen), so
  the corpus is 840 rather than 1000 episodes. Re-running the missing stripe
  indices extends it reproducibly; contributing shards are listed per split in
  MANIFEST.json.

INFORMATION BOUNDARY (the thing reviewers will attack): ALFWorld's
`admissible_commands` and both domains' expert plans are COLLECTION labels
only. Deployment scores a separately generated grounded catalogue
(`observed-entities-v1` for ALFWorld: 35-2895 actions/step; formable block
operations for PlanBench: 18-50). Collection FAILS HARD if that catalogue
omits the expert action, pinned by
`tests/test_alfworld_adapter.py::test_collection_fails_when_grounded_catalogue_omits_expert_action`,
so 100% catalogue recall is an invariant, not a survivor average.

ADMISSION GATES — both domains pass every DATA-side gate, both FAIL the
tiny-overfit gate. PlanBench (`_gate_planbench_v2`, 6/8 checks):
schema/replay/recall/disjointness pass (exact replay 1.0, catalogue recall
1.0, goal success 1.0 on ALL 3000+60+80 episodes); horizon Energy live;
random 0.0 vs oracle 1.0; dropout zero + EMA eval pass; closed loop runs on
both interfaces. But a 0.36M model trained 400 epochs on 24 episodes reaches
strict success 0.0 on its own TRAINING set (random 0.0), while memorising the
data well (`observed_action_sequence_exact` 0.702, token acc 0.950,
`goal_dist_corr` 0.579) and emitting invalid actions 92% of the time on the
full catalogue. Length-2 control (`_gate_planbench_len2`, 7/8): with only
2-step plans and near-perfect geometry (`goal_dist_corr` 0.995) the model gets
0.167 vs RANDOM 0.250 — i.e. the failure is NOT plan-length compounding, it is
that the endpoint-Energy selection does not beat random on this domain at tiny
scale. The shuffle falsifier IS wired and bites hard at the loss level
(sequence-exact 0.702 aligned vs 0.013 shuffled); it only scores "false" in
the summary because that check compares strict success, which is at floor for
both cells. ALFWorld (`_gate_alfworld`): data gates pass (exact replay 1.0,
recall 1.0, goal success 1.0 on 40 sampled episodes/split — full live-engine
replay of 1000 episodes is hours), horizon Energy live at h=1 and h=8,
random 0.0 vs oracle 1.0; learning-side cells (40 epochs) were still running
when this entry was written — see `_gate_alfworld/gate_summary.json`.

CONSEQUENCE: per the admission rules in PAPER_EXPERIMENTS.md neither domain is
admitted for the LR sweep yet. Both first cells are PREPARED, NOT LAUNCHED:
`2026-08-12-intent-domains-v1/{planbench,alfworld}-ldad-lr3e4-s0-v1/job.sh`
(LDAD, lr 3e-4, seed 0, width 256, `state=PREPARED`, snapshot under
`runs/autonomy/_code/<sha>`). Open question for the owner: whether the
tiny-overfit criterion is the right admission bar for domains whose plans are
7-15 steps long, or whether it should be scored on a non-floored metric
(LDAD sequence-exact / goal_dist_corr), which both domains pass clearly.

Gate-tooling fixes committed this round: `TINY_DATA_ROOT` so the tiny cells can
use a cheap subset while replay/bounds still read the full corpus; and
`check_compiled_domain_horizon_loss.py` no longer hardcodes iGSM's
`max_chunk_len=96`, which had made the ALFWorld gate impossible to run at all.

## 2026-08-12 (cont. 4): monitoring sweep advances

- Predictor variants COMPLETE (report 2026-08-12-predictor-variants/,
  mirrored): non-residual strict@16 .980, causal .970 vs headline .975 —
  recipe is predictor-agnostic; ablation-table entry. Task #7 closed.
- Faithful screen interim: 3e-4 strict@16 .243/slack4 .707 beats 1e-3
  (.210/.563). NOTE: depth curve is FLAT on faithful (d1 .263 -> d16 .243)
  — planning depth buys nothing on faithful iGSM so far; hold
  interpretation until screen + 5 seeds. Shuffle arm running; 1e-4, 3e-3
  queued.
- sent-lm-long-s0 (15-25 steps) COMPLETED early (ran in first chain):
  loops1-decoder success recorded in its metrics.json; note current chain
  may rerun it after tok-lm — harmless overwrite, same config.
- Alex: 12 running / 49 pending (mains + ProofWriter screen).
