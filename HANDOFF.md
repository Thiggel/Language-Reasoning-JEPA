# HANDOFF — complete state of the TextJEPA project

_Last updated: 2026-07-14._

**Current directive overrides the historical in-flight section below.**
Hard-text and all non-hierarchy work remain paused. Continuous HWM-style
planning remains negative (.289 versus .324 flat). Wave 11 finds exact
controller-outcome reranking headroom (.44 versus .26), but learned rerankers
do not recover it. A discrete text-span planner initially appeared positive,
but its H2/H3 confirmations are invalid: lexicographic sequence truncation
over-represented early first actions. Root-balanced arbitrary planning falls
to .110 versus .310 flat; learned-support beam plus one-step value reaches
.250. The implementation now balances every proposal bank over the executed
first action, randomizes future proposals reproducibly, and reports an
always-first baseline. Faithful iGSM had a second ordering leak: the reference
parameter order makes always-first solve 100/100 long problems, so every model
now receives a stable problem-specific shuffled menu. Planning-matched support
plus prefix-aware macro value reaches .320--.330 on the selection screen only
with a flat one-step safeguard. Its frozen corrected 3×500 confirmation is
consistently negative: .310/.326/.296 versus flat .318/.332/.300 (means .311
versus .317). No hierarchy is selected; see
`research/intent_phrase/waves/11_controller_outcomes_and_discrete_hierarchy.md`.
This document is the single entry point for continuing this project. Read it
together with `RESULTS.md` (all findings, numbered) and
`reports/discourse_jepa_neurips.pdf`, the authoritative paper-facing deck.
`reports/discourse_jepa.pdf` is retained only as a labeled historical
chronology; do not copy its old artifact vocabulary or headline tables into
the paper.
The paper-facing deck includes the ordering correction and final paired
confirmation. Recompile and visually check the canonical
`reports/discourse_jepa_neurips.pdf` after any further edits.

---

## 1. What this project is

Four controlled reconstruction-free latent world-model variants over language:

- **Deterministic observed-action discourse** (`disc_*` runs): a reasoning
  trace is a trajectory —
  state `s_t` = compressed discourse state, action = *intent phrase* (never
  contains the outcome), predictor `F(s,a)` must compute consequences in
  latent space. Generation = closed-loop MPC over the environment's
  discrete feasible-action interface, scored by a learned energy.
- **Edit track** (`edit_*` runs): state = text buffer (draft solution),
  actions = delete/insert/replace edits, "edit until perfect".
- **Observed-action probabilistic discourse** (`dvjepa_*`, `dvldad_*`): the
  intent phrase remains observed while the next latent state is Gaussian.
  Raw-token displacement decoding is faithful LDAD in this track.
- **Action-free probabilistic sentence stream** (`sentence_vjepa_*`,
  `svjepa_*`): prompt and solution sentences are packed into one causal stream
  with no retained
  intent/outcome boundary. Both the next latent state and the unobserved
  transition code are diagonal-Gaussian variables. There is no intent input,
  symbolic action target, feasible-action interface, or token decoder.

Domains, in increasing difficulty/faithfulness:
1. **Stylized iGSM** (`data=igsm`): our generator, mod-23 DAGs, 3–9
   necessary steps. Where all recipe science was done.
2. **Hard stylized** (`data=igsm_hard`): 10–18 steps — the depth frontier.
3. **Faithful iGSM** (`data=igsm_real`): the OFFICIAL facebookresearch/iGSM
   generator (vendored, MIT, `third_party/iGSM`), adapter in
   `src/textjepa/data/faithful.py`. All text produced by the reference
   renderer; validated by the official checker (`scripts/
   validate_faithful.py`: 60/60 med, 99/100 hard — the one hard failure is
   a documented checker operand-order corner). Med = max_op 15/max_edge 20;
   hard = 21/28.

### Terminology used in all new tables

- **Strict budget**: an episode may execute exactly the number of actions in
  an optimal necessary-only solution. A distractor is not assigned a special
  training label by this metric; it consumes one of the available actions, so
  the planner cannot finish before evaluation stops.
- **Slack-2 budget**: the planner receives `n_necessary + 2` executions. A
  successful episode can therefore contain at most two extra actions. The
  evaluator stops at the budget; it does not let the planner continue and
  relabel a longer solution as wrong afterward.
- **GAR H/K**: H is the number of environment transitions used by the
  geometry-greedy continuation teacher; K is the number of alternative root
  actions (plus the demonstrated action). H=2/K=2 does *not* denote the
  hierarchy or open-loop rollout objectives. Those are independent losses and
  are shown in separate ablation columns.
- **Trace monotonicity**: the clean (`label_free`) hinge asks every observed
  transition to reduce latent distance to its trace-terminal state. It does
  not assert that the symbolic/oracle goal distance is monotone and does not
  read relevance labels. The legacy form did use necessary/distractor flags.
- **Depth-calibrated energy** (historically “cross-horizon calibrated”): a
  symbolic legacy loss compares complete `depth + value` costs across search
  depths. It is distinct from H-step GAR and is excluded from the clean model.
- **Hybrid displacement labels**: the old displacement head predicted a
  symbolic operation class plus an EMA intent embedding. “Retaining hybrid
  labels” meant this head was accidentally left in an otherwise clean K=2
  pilot. Current clean runs set it to zero. Faithful LDAD instead reconstructs
  every observed intent token from `s_{t+1}-s_t` and uses no operation label.

## 2. Environment / infrastructure

- **Venvs**: `.venv` (training; system CPython — its pyexpat is broken, do
  NOT import matplotlib here) and `.venv2` (uv-managed; matplotlib, LaTeX
  figure work, also used by `scripts/eval_run.sh` via `PY=`). Both need
  `transformers` (the vendored iGSM tokenizer imports it) — already
  installed in both.
- **GPUs**: local box (2×V100 + RTX6000) + gruenau1–12
  (`ssh laitenbf@gruenauN.informatik.hu-berlin.de`, same NFS home; helper
  fns `gruenau`/`gruenau-gpus` in `~/.bashrc`/`~/.zshrc` list free GPUs).
  gruenau12 = 10×L40 (usually freest), 9/10 = A100s, 11 = H100s.
- **Detached remote jobs** (the only reliable pattern — note the `cd`
  INSIDE the quoted command and the `timeout` around ssh):
  ```bash
  timeout 25 ssh -o BatchMode=yes laitenbf@gruenau12.informatik.hu-berlin.de \
    "setsid nohup bash -c 'cd /vol/home-vol2/ml/laitenbf/TextJEPA; \
     export CUDA_VISIBLE_DEVICES=N; <chain>' >/dev/null 2>&1 < /dev/null &"
  ```
- Logs: `runs_<name>.log` at repo root (append). Artifacts:
  `runs/<name>/{best.pt,last.pt,metrics.csv,probe_results.csv,probe_v2.csv,
  plan_*.json,counterfactual_audit.json,...}`. `runs/` is git-ignored.
- GitHub: `git@github.com:Thiggel/Language-Reasoning-JEPA.git` (main).

## 3. How to run things

```bash
# train (hydra; every experiment is a file in configs/experiment/)
.venv/bin/python scripts/train.py +experiment=<name>
# standard eval bundle: probes + plan slack0/2 + geometry plots
PY=.venv2/bin/python bash scripts/eval_run.sh runs/<name> cuda:0
# information-matched planning (exhaustively ranks all currently feasible intents)
.venv2/bin/python scripts/plan.py ckpt=runs/X/best.pt slack=0 lookahead=1
# diagnostic only: depth > 1 uses the reference graph to enumerate future
# feasible actions and must be explicitly enabled/labeled
.venv2/bin/python scripts/plan.py ckpt=runs/X/best.pt slack=0 lookahead=2 \
    allow_oracle_future_actions=true
# counterfactual audit (matching / Kendall-tau / RSA vs symbolic truth)
.venv/bin/python scripts/audit_counterfactual.py ckpt=runs/X/best.pt n_episodes=100
# LM baselines: scripts/train_lm.py + scripts/plan_lm.py
# target_kind=intent ranks the same intent phrases as JEPA; outcome is an
# oracle-candidate diagnostic because rendered candidates contain values.
# Candidate intent strings use mean per-token likelihood/CE by default, so
# variable wording length cannot determine the ranking.
# The sentence-LM scripts support the same outcome/intent distinction.
# variational planner: scripts/plan_var.py (samples codes from the prior)
# faithful fidelity gate: .venv/bin/python scripts/validate_faithful.py 60
# guarded final test (only after selection is frozen):
FINAL_TEST_CONFIRM=recipe-frozen scripts/eval_final_test.sh runs/X latent cuda:0
# figures + report: .venv2/bin/python scripts/make_report_figs.py; scripts/report.py
# paper-facing deck: cd reports && pdflatex discourse_jepa_neurips.tex (twice)
# discourse_jepa.tex/pdf is a labeled historical chronology, not the deliverable
# tests: .venv/bin/python -m pytest -q  (51 tests, all green)
```

Known hydra pitfalls: keys that already exist in a config must be set
WITHOUT `+` prefix; experiment files with a `defaults:` list need it as the
first block after the `# @package _global_` header.

## 4. Where the science stands (current information-matched numbers)

Stylized validation success at lookahead 1, strict / slack-2. All principal
baselines and the current H=2 JEPA reference now have three seeds; the recipe
is not frozen because the bounded-continuation selector remains active.

| model | strict | slack-2 |
|---|---:|---:|
| random feasible policy | 0.055 | 0.405 |
| token intent policy (3 seeds) | **0.827 ± 0.003** | **0.978 ± 0.003** |
| sentence intent policy (3 seeds) | 0.690 ± 0.053 | 0.903 ± 0.039 |
| sentence policy + next-latent likelihood (3 seeds) | 0.817 ± 0.034 | 0.953 ± 0.010 |
| sentence policy + latent-distance selection (3 seeds) | **0.840 ± 0.017** | 0.965 ± 0.013 |
| reduced non-symbolic JEPA (3 seeds) | **0.797 ± 0.008** | 0.963 ± 0.008 |
| symbolic-preference diagnostic (3 seeds) | 0.962 ± 0.013 | 0.997 ± 0.003 |
| oracle | 1.000 | 1.000 |

The reduced JEPA uses direct one-step latent-state prediction,
on-trajectory outcome prediction with its internal predicted-outcome rollout,
variance--covariance regularization, an EMA target, and H=2/K=2 multi-step
latent-goal preference distillation. It does not use symbolic ranking,
terminal monotonicity, hierarchy, the separate open-loop latent objective,
counterfactual transition prediction, a residual state skip, scalar value
regression, or LDAD. Exact two-seed add-back tests rejected both scalar value
regression and faithful raw-token LDAD in this reduced model. All current
feasible intents are enumerated for JEPA and LM policies.

Do not quote historical lookahead-2/4 results as model-only planning. The
implementation must use the reference dependency graph to enumerate future
feasible actions and identify terminal sequences. Those runs are now guarded
by `allow_oracle_future_actions=true`, labeled as oracle-action diagnostics,
and excluded from headline comparisons. Legacy hard-domain and edit-track
results remain in `RESULTS.md` but have not yet been rebuilt under the frozen
current protocol.

## 5. IN FLIGHT right now (checked 2026-07-14, 07:48 CEST)

Current decision-critical jobs: the faithful-LDAD reduced-model add-back is
complete and rejected; its two-seed mean is 0.788/0.958 versus 0.798/0.968
without it. The reduced H=4/B=8 bounded-beam student is at epoch 6/20, and the
H=8/B=8 student is at epoch 2/20. The H=16/B=8 teacher-only audit is complete at
0.895 top-1/0.672 tau-a and is not promoted because it does not improve on
H=4/B=8. The H=2 reduced reference is complete over three seeds at
0.797 ± 0.008 / 0.963 ± 0.008 (exact 0.805/0.790/0.795 and
0.970/0.965/0.955). The exact-paired shuffled-intent action-grounding
falsifier is active. An earlier permutation consumed the
RNG before GAR-anchor sampling and was therefore only distribution-matched.
The dataset now samples the complete geometric teacher first, and a
regression test proves equality of prompt, trace, anchor, alternatives, and
outcomes across aligned/permuted conditions.
`scripts/report_action_grounding.py` reports only this exact-paired control in
`runs/action_grounding.md`. That corrected control is now active on gruenau11
GPU 0 and is at epoch 7/20. The completed distribution-matched diagnostic gives 0.660/0.900 versus
0.805/0.970, transition match 0.350 versus 0.997, value tau 0.778 versus
0.904, and clean/off-history top-1 0.887/0.646 versus 0.940/0.706. This is
strong preliminary evidence, but only the exact-paired rerun is paper-grade.
The
symbolic-preference reference is complete over three
seeds at 0.962 ± 0.013 / 0.997 ± 0.003. The stylized token policy is now
complete over three seeds at 0.827 ± 0.003 / 0.978 ± 0.003 (exact strict
0.830/0.825/0.825, slack-2 0.980/0.980/0.975). The plain sentence baseline is complete over three
seeds at 0.690 ± 0.053 / 0.903 ± 0.039, and the sentence-plus-latent baseline
is complete at 0.817 ± 0.034 / 0.953 ± 0.010 by
likelihood and 0.840 ± 0.017 / 0.965 ± 0.013 by latent distance; the live
baseline report never replaces a seed-1 number with a partial-seed mean.
The exact reduced-model residual-predictor add-back is active; it changes only
`predictor_residual=false` to `true` and closes the requested direct-versus-
residual architecture control around the selected objective set.
The matched seed-3 faithful-LDAD add-back is also active against the live
seed-3 reduced reference, completing a three-seed paper-facing displacement
control without changing the already rejected two-seed selection decision.
The third matched removal of the outcome head's internal predicted-outcome
consistency term is active as `disc_latent_goal_h2_r2_nooutroll_s3` on
gruenau7 GPU 1. Together with the active third outcome-anchor removal, these
complete the two borderline final mechanism rows while the selector runs;
the obviously large-effect removals wait until the continuation teacher is
frozen so they are not replicated around a potentially superseded reference.
`scripts/report_ablation_matrix.py` writes the live seed-matched mechanism
table to `runs/component_removal_matrix.md`; it reports the exact number of
matched pairs and refuses to imply that a one- or two-seed row is a final
three-seed paper result.
Official plain-sentence, sentence-plus-latent, and token-policy seeds 2--3 are
all active on recycled GPUs. Official symbolic-reference seeds 2--3 are now
active; seed 3 was launched on gruenau9 GPU 1 when the action-free control
finished.
The clean official deterministic transfer remains active. The official
action-free posterior-code reconstruction control and its same-state audit are
complete. The controlled official observed-action architecture pairs are also
complete. The ordered-token EMA+SIGReg pair is negative relative to mean
pooling; the pooled direct-prediction EMA+VICReg pair confirms the LDAD
transition-grounding effect. Its completion released gruenau9 GPUs 0/2, which
now run the missing official EMA+VICReg residual pair with raw-action LDAD
off/on. This holds the best-fidelity predictor fixed and will select the
official observed-action probabilistic regularizer. The faithful-domain build-up has
also started in parallel around the imported reduced model: exact additions
of raw-action LDAD, residual transition parameterization, the separate
open-loop latent-prediction loss, hierarchy, counterfactual-transition
augmentation, and label-free geometry-to-energy regression are all active.
The live component table is
`runs/official_recipe_screen.md`. The predeclared addition rule is recorded
in `runs/official_build_decision.md`: an addition must improve at least one
budget by more than 0.02, degrade neither by more than 0.02, and receives a
matched second seed when its best one-seed gain is 0.02--0.05. Larger gains
advance to a combined candidate immediately; after replication, a mean gain
above 0.02 advances without requesting further seeds. The matched one-seed official
autoregressive baselines are complete: random 0.200/0.460, token likelihood
0.415/0.675, sentence likelihood 0.450/0.755, sentence+latent likelihood
0.455/0.765, and sentence+latent distance 0.445/0.715. Preference-margin and
scalar-value replications are complete and rejected, and the greedy H=16
control is complete and negative.

The 13-run random-shooting GAR/counterfactual screen and the corrected
18-cell sentence-stream factorial are complete. Main screening results:

| model | strict / slack-2 |
|---|---|
| fresh scalar + hybrid-displacement baseline | 0.705 / 0.950 |
| matched symbolic ranking K=2 | **0.960 / 0.995** |
| counterfactual outcomes K=1/2/4/8 | 0.735/0.940, 0.735/0.960, 0.720/0.945, 0.715/0.930 |
| clean counterfactual K=2 (no GAR) | 0.180 / 0.595 |
| hybrid-label GAR H=2/4/8 | 0.690/0.915, 0.685/0.915, 0.645/0.915 |
| clean CF+GAR H=2/4/8, K=2 | **0.705/0.880**, 0.655/0.920, 0.650/0.875 |
| clean CF+GAR H=4, K=4/8 | 0.670/0.905, 0.670/0.910 |

The old clean CF+GAR configs explicitly disabled hierarchy and open-loop
rollout. They must not be described as containing those objectives. H is the
GAR label horizon, not the learned hierarchy or rollout loss.

The variational 18-cell conclusion is separate: no fully online-gradient
cell has both healthy scale and high state rank. `online_sg + SIGReg` without
posterior-code reconstruction is the best no-EMA diagnostic (state std/rank
0.991/136.0); adding that reconstruction lowers rank to 117.9 and the
posterior-mean action rank from 15.6 to 11.7. The inferred target is vulnerable
to collusion; this is neither observed-action Delta-JEPA nor faithful LDAD.
The complete generated table is `runs/variational_factorial.md`; always read
state scale and effective rank together (several nominally high-rank cells
have state std near 0.001 and are collapsed in magnitude).

New code now provides (a) faithful text LDAD, reconstructing the raw observed
intent tokens from `s_{t+1}-s_t`, and (b) geometry-greedy N-step GAR. At each
continuation state greedy GAR enumerates all feasible actions, executes them,
EMA-encodes their true outcome text, and follows the child closest to the
terminal goal. It never reads remaining counts, ancestors, relevance flags,
or symbolic action quality.

The original matched recipe screen is complete for H=1/2/4/8/16 and the full
first-round one-component table. These older full models retained hierarchy
and open-loop rollout so that their removals could be isolated:

| group | runs |
|---|---|
| greedy GAR + faithful LDAD | `disc_gar_greedy_h2_k2_{noldad,ldad}` plus LDAD weights 0.05/0.2/1.0 |
| greedy-vs-random control | `disc_gar_random_h2_k2_{noldad,ldad}` |
| greedy horizon | `disc_gar_greedy_h{1,2,4,8,16}_k2_ldad` |
| no-EMA/no-stopgrad control | `disc_gar_greedy_h2_k2_{noldad,ldad}_online` |

All configurations compose, CPU and GPU smoke tests pass, and the full suite
is 40/40 green under `.venv/bin/python -m pytest -q`. K=4/K=8, H=16, and the token-order-preserving action-encoder
diagnostics are complete. H=16 reaches **0.655 strict / 0.895 slack-2**,
teacher top-1 0.790, transition matching 0.912, and value-order tau 0.768;
clean/after-error useful-action top-1 is 0.870/0.568. It confirms that longer
width-one continuation is worse than H=2. That encoder
projects every intent token to 8 dimensions, concatenates positions, and then
uses the same 16-d action bottleneck as the pooled encoder. Do not launch the
three-seed paper matrix until the fixed-point shear below selects the recipe.

The first newly completed control is random-shooting H2, K2, no faithful
LDAD, with hierarchy/rollout enabled: **0.670 strict / 0.895 slack-2**;
counterfactual audit matching 0.743 and value Kendall tau 0.833. The older
otherwise-matched clean run without hierarchy/rollout was 0.705/0.880,
matching 0.868, tau 0.812. This does not isolate either auxiliary, but gives
no evidence that adding both helps. Their exact greedy-H2 removals are
complete and both auxiliaries are absent from the reduced model.
The exact faithful raw-text LDAD counterpart is now complete at
**0.720/0.940**, matching 0.974, value tau 0.830. Relative to LDAD-off this is
+0.050/+0.045 planning, +0.231 transition matching, and -0.025/-0.039
distractor rate (strict/slack-2). This is promising one-seed evidence at the
predeclared second-seed boundary, not a final LDAD claim. The new direct
GAR-teacher audit shows that the H2 labels are 95.9% correct when decisive but
cover only 73.3% of oracle-distinct pairs and spend about 21% of emitted
preferences on oracle ties. LDAD does not change the teacher algorithm or
consume oracle labels, but it can change the learned target geometry and thus
the emitted preferences. Older LDAD-off/on audits used unequal sample counts,
so do not quote them as a paired teacher-effect estimate. `eval_run.sh` now
adds a consistently configured audit to every GAR evaluation.
The exact no-counterfactual-prediction removal is complete at
**0.755/0.935**, matching 0.988 and value tau 0.858. Its local value top-1 is
0.927; over the mean 4.07 necessary decisions, $0.927^{4.07}=0.735$, close
to observed strict success. This explains much of the remaining episode gap
as compounding local ranking error. The final epoch-19 H2 reference checkpoint
plans at **0.750/0.940**; its comprehensive probe/audit chain is still
completing, but the planning measurement uses the final frozen checkpoint and
the standard deterministic 200-episode evaluator.
The no-continuation control H=1/K=2 is complete at only
**0.315 strict / 0.800 slack-2**, matching 0.887 and value tau 0.542. Its
teacher covers 55.2% of oracle-distinct pairs, reaches 0.879 accuracy when
decisive but only 0.750 oracle top-1, and the student value top-1 is 0.726.
Thus H=2 does not denote hierarchy or open-loop prediction, but at least one
actual continuation step in the geometric preference teacher is necessary.
The first longer geometry-greedy member, H=4/K=2, reaches
0.695/0.930, matching 0.883 and value tau 0.792. Its teacher has oracle top-1
0.900 overall, but only 0.722 on traces containing distractors (pair tau-a
0.228, versus 1.000/0.898 on clean traces). Longer continuation therefore
does not automatically repair the trajectory-specific terminal-goal proxy.
H=8/K=2 degrades further to **0.620/0.885**, matching 0.901 and value tau
0.728. Teacher coverage rises to 0.921, but decisive pair accuracy falls to
0.836 and oracle top-1 to 0.830. The controlled curve therefore peaks at H=2:
H=1 is too myopic, while longer geometry-greedy continuations accumulate
teacher error and increase the distractor rate.
The matched fixed-checkpoint width-one audit reaches the same conclusion without
student-retraining confounds. Recomputing the current reduced checkpoint's
teacher at H=1/2/4/8/16 gives oracle top-1
0.70/0.86/0.81/0.80/0.80 and pairwise tau-a
0.354/0.720/0.677/0.616/0.604. Pair coverage does not explain the decline:
it is 0.768/0.841/0.884/0.848/0.884. H=2 is therefore the width-one teacher
optimum for the present geometry as well as the completed end-to-end optimum.
A new bounded-beam diagnostic shows that greedy continuation error explains
part of the longer-horizon decline. On the same 200 anchors, H=4 with beam
width B=1/2/4/8 gives teacher top-1 0.810/0.830/0.870/0.900 and pairwise
tau-a 0.646/0.641/0.701/0.728; the paired H=2/B=1 reference is 0.865/0.661.
H=8 likewise improves from 0.805/0.580 at B=1 to 0.900/0.670 at B=8. Beam
search uses no symbolic quality signal and costs O((K+1)HBA), not
O((K+1)A^H), where K is the number of alternatives and A is mean branching.
H=16/B=8 reaches 0.895/0.672 with coverage 0.945, no better than the H=4/B=8
teacher despite twice the horizon, so it remains audit-only. Exact reduced-
model H=4/B=8 and H=8/B=8 student cells are active; replicate a cell only if
planning improves. Because B=8 adds substantial training interaction and
compute, a gain of at most 0.02 on both planning budgets selects the simpler
H=2/B=1 teacher; a 0.02--0.05 gain gets a matched second seed, and a larger
gain is retained immediately. The teacher audit is in
`runs/preference_sweep.md`; the executable planning gate is
`scripts/selector_decision.py` and writes `runs/selector_decision.md`.
The matched root-candidate sweep also selects K=2. At fixed H=2,
K=2/4/8 gives strict success **0.750/0.740/0.710**, slack-2
0.940/0.935/0.910, teacher top-1 0.930/0.920/0.890, pair coverage
0.836/0.743/0.696, and teacher tau-a 0.776/0.694/0.615. K=8 retains strong
transition matching 0.988 but does not improve selection. More alternative
roots add progressively noisier geometric comparisons; K=2 is sufficient.
The zero-margin monotonicity control is also complete at 0.765/0.940. Full
removal is slightly better at 0.780/0.950, so there is no evidence for
retaining trace-terminal monotonicity.
At seed 1, seven exact removals satisfied the predeclared two-metric removal
rule: LDAD 0.755/0.960, monotonicity 0.780/0.950, value distillation
0.800/0.920, counterfactual transition prediction 0.755/0.935, hierarchy
0.765/0.940, rollout 0.840/0.970, and the residual transition skip
0.800/0.960. Value distillation is no longer a settled removal: at seed 2 the
reference reaches 0.785/0.960 while its exact removal falls to 0.680/0.945,
reversing the strict effect. The two-seed mean loss is 0.0275 strict / 0.0175
slack-2, so it was retested as an exact add-back in the reduced model rather
than spending another run on the obsolete over-complete recipe. That add-back
is now rejected: its two-seed mean is 0.750/0.960 versus 0.798/0.968 without
the scalar target. The matched
no-LDAD seed-2 cell is now complete at **0.720/0.920**, versus
**0.785/0.960** with faithful LDAD. Across the two matched seeds, the
LDAD/no-LDAD means are 0.768/0.738 strict and 0.950/0.940 slack-2: faithful
raw-intent reconstruction contributes +0.030/+0.010 on average, but with a
sign reversal across seeds. It must therefore not be called removable from
that over-complete model. The exact reduced-model LDAD add-back is complete at
0.770/0.955 and 0.805/0.960
across seeds, a mean 0.788/0.958 versus 0.798/0.968 without it. Because
removing it improves both budgets by 0.010, faithful LDAD is excluded from the
reduced planner even though it strongly improves isolated transition
grounding. The exact EMA removal falls to **0.575/0.840**, with transition
matching 0.943 and value tau 0.735, so EMA is retained. The formal combined
round-one model is `disc_latent_goal_h2_r1`: direct next-state prediction,
one-step latent prediction, on-trajectory outcome prediction, VICReg, EMA,
and H=2/K=2 geometric preference distillation. It has no symbolic ranking,
LDAD, monotonicity, scalar value distillation, counterfactual transition
loss, hierarchy, open-loop latent loss, or residual skip. Its first two
selection seeds are complete. The seed-1 validation planning result is
**0.805 strict / 0.970
slack-2**, improving the over-complete reference by 0.055/0.030. Its exact
seed-2 reference reaches **0.790/0.965**, transition matching 0.999, and
clean/after-error useful-action top-1 0.934/0.758. The matched two-seed mean
used for the seed-1/2 outcome-removal decisions is 0.798/0.968. The third
reference seed reaches 0.795/0.955, giving a three-seed mean of
0.797 ± 0.008 / 0.963 ± 0.008. This is not yet the selected paper model: the matched token policy reaches
0.827 ± 0.003 / 0.978 ± 0.003 over three seeds and the completed
three-seed sentence-plus-latent policy reaches
0.840 ± 0.017 / 0.965 ± 0.013 with latent-distance selection. Its transition
audit gives matching **0.997**, value-order tau
**0.904**, and local value top-1 **0.948**. Compounding local top-1 across the
mean 4.07 decisions predicts $0.948^{4.07}=0.804$, essentially the observed
0.805 strict success. The matched closed-loop audit against the fresh
symbolic-preference reference localizes the gap: counterfactual transition
matching is 0.997 vs 1.000, but useful-action top-1 on clean histories is
0.940 vs 0.989, after-error top-1 is 0.706 vs 0.857, and the median
useful-vs-distractor energy margin is 0.610 vs 1.617. Only 1.2% of the clean
model's competitive decisions have absolute margin below 0.02. Failure rate
grows from 0.084 at 3--4 necessary steps to 0.373 at 5--6 and 0.600 at 7--9;
all failed strict episodes contain a distractor action. The residual gap is
therefore sharper action ordering and off-trajectory robustness, not
transition identity or tie-breaking. The detailed records are
`runs/{disc_latent_goal_h2_r1,disc_fresh_symbolic_rank_k2}/failures.json`.
Its exact second-round removals test direct latent prediction, frozen outcome
prediction, VICReg, geometric preferences, and EMA. A sixth control removes
only the predicted-outcome rollout internal to the outcome-prediction loss.
All six are now complete: every component is retained, including both the full
outcome target and its internal rollout by matched two-seed tests.
The live component-explicit table is `runs/round2_screen.md`; it reports the
latent, outcome, outcome-rollout, preference, VICReg, and target-network
switches directly rather than relying on artifact names.
The residual selector gap is also being tested without adding supervision:
`runs/selector_screen.md` compares preference margins 0.25/0.5/1.0/2.0, a
smaller geometric label filter (0.005 vs 0.02), doubled distractor exposure
(0.30 vs 0.15), and geometry-to-value regression. These interventions
separate insufficient energy separation, noisy-label filtering, and poor
off-trajectory coverage. The smaller 0.005 label-gap result is complete and
worse: strict/slack-2 falls **0.805/0.970 -> 0.740/0.925**, value tau falls
0.904 -> 0.874, clean-history useful-action top-1 falls 0.940 -> 0.914, and
the median useful margin falls 0.610 -> 0.520. The reference 0.02 filter is
retained: geometric differences below it add noisy preferences rather than
useful coverage. Do not select among the remaining interventions before their
frozen planning and failure audits complete. The doubled-distractor cell is
now complete and is also worse: strict/slack-2 is **0.790/0.935**, transition
matching 0.995, value tau 0.882, and clean/after-error useful-action top-1
0.933/0.649. Every corresponding reference value is higher
(0.805/0.970, 0.997, 0.903, and 0.940/0.706). More off-trajectory examples at
this mixture weight therefore do not solve recovery and are rejected.
The preference-margin sweep and matched soft-margin replication are complete.
At seed 1, default margin 0.5 scores 0.805/0.970 and margin 0.25 scores
0.800/0.985, while margins 1.0 and 2.0 fall to 0.575/0.870 and 0.435/0.875.
The apparent soft-margin gain does not replicate: at seed 2, margin 0.25
scores **0.725/0.950** versus **0.790/0.965** for the matched reference.
Two-seed means are therefore **0.763/0.968** for margin 0.25 and
**0.798/0.968** for margin 0.5. The 0.035 strict loss exceeds the replicated
retention threshold, so the default 0.5 margin is retained. At seed 2 the
softer margin trades clean-history top-1 for recovery (0.900/0.776 versus
0.933/0.758 for clean/after-error decisions), which explains why better
off-history accuracy does not translate into strict episode success.
The completed closed-loop audits explain why larger hinges fail: margins
1.0/2.0 increase the median useful-action energy gap to 0.920/1.477, but
clean-history top-1 falls to 0.844/0.749. They create large, incorrectly
ordered separations rather than calibrated confidence.
The exact reduced-model geometry-to-value add-back replication is complete
and rejects the scalar target. Seed 1 scores 0.760/0.950 versus 0.805/0.970;
seed 2 scores **0.740/0.970** versus **0.790/0.965**. Two-seed means are
**0.750/0.960** with value regression and **0.798/0.968** without it. The
0.048 strict loss is decisive; mean transition matching is still 0.999 but
value-order tau is only 0.844. Scalar geometry-to-value regression remains
absent from the reduced recipe. At seed 2 it again trades clean accuracy for
recovery (clean/after-error top-1 0.913/0.781 versus 0.933/0.758), which does
not improve whole-episode success.
The first second-round endpoint is complete: removing geometric preference
distillation gives **0.115 strict / 0.500 slack-2**, transition matching
0.979, and value-order tau 0.080. The dynamics remain grounded while the
selector is almost random (random 0.055/0.405), so the preference objective is
load-bearing. Relative to the reduced reference, the exact losses are
0.690 strict / 0.470 slack-2, so this component is retained.
The exact latent-state-prediction removal is also decisive: it reaches only
**0.495/0.825**, with transition matching 0.265 (chance 0.287), value-order
tau 0.671, and oracle-trace top-1 0.854. This objective is retained; it is what
grounds the action-conditioned transition, whereas the preference loss turns
grounded transitions into a policy.
The outcome-target branches affect a different mechanism. Removing all
on-trajectory outcome prediction gives **0.755/0.945**, matching 0.999 and
value tau 0.875. Removing only its internal predicted-outcome rollout gives
**0.770/0.935**, matching 0.998 and tau 0.886. Their closed-loop clean/after-
error useful-action top-1 values are 0.922/0.630 and 0.927/0.584, versus
0.940/0.706 for the reference. They improve energy ordering and recovery from
off-trajectory states, not transition identity. Their seed-1 losses
(0.050/0.025 and 0.035/0.035 strict/slack-2) fell in the predeclared
second-seed band, so both were replicated against one shared seed-2 reference.
Across the two seeds, removing only the predictor's internal predicted-outcome
rollout averages **0.772 strict / 0.933 slack-2**, versus 0.797/0.968 for the
reference. The mean losses are therefore 0.025/0.035, above the predeclared
0.020 removal threshold. Retain this rollout term. The broader outcome-target
pair has now completed as well: removing the full target averages
**0.750/0.942**, giving 0.048/0.025 losses against the same reference. Retain
the on-trajectory outcome target too. Both results preserve transition
matching (0.998--1.000) while reducing value-order tau and planning, confirming
that these terms organize action-selection energy rather than identify the
one-step transition.
The on-trajectory outcome-prediction removal is the first retained component:
it falls to **0.680/0.940**, a 0.070 strict loss, even though transition
matching is 0.998. Its value tau falls from 0.847 to 0.800. Retain this loss in
round one; it helps selection-energy organization rather than raw transition
identity.
VICReg is also decisively retained: removing it collapses planning to
**0.455/0.765** (losses 0.295/0.175), transition matching to 0.765, and value
tau to 0.680. Faithful LDAD is present in this cell, confirming at full-model
scale that displacement reconstruction does not replace distributional
regularization or learned-energy calibration.
The exact reduced-model removal confirms the decision without the old
auxiliaries: strict/slack-2 falls from **0.805/0.970** to **0.615/0.755**,
transition matching falls 0.997 -> 0.953, and value-order tau falls
0.904 -> 0.790. Clean-history useful-action top-1 is 0.870, but after an
earlier distractor it is only 0.336. VICReg therefore contributes to both
transition organization and selector recovery in the selected architecture;
it is not retained merely because of a marginal state-variance statistic.
EMA is likewise decisively retained in the reduced architecture. Replacing
the target network by an online stop-gradient target changes strict/slack-2
from **0.805/0.970** to **0.720/0.910**, transition matching from 0.997 to
0.987, value tau from 0.904 to 0.871, and clean/after-error useful-action
top-1 from 0.940/0.706 to 0.908/0.562. The 0.085/0.060 losses exceed the
one-seed retention boundary, so no second seed is required for this decision.
The residual transition skip is removed. Direct next-state prediction reaches
**0.800/0.960**, versus 0.750/0.940 for
`s_hat(t+1) = s_t + F(s_t,u_t)`; transition matching/value tau also improve
from 0.986/0.847 to 0.995/0.891. This is an exact one-component result, not an
architectural assumption carried forward from the original implementation.
On a larger 500-anchor audit, the trace-terminal teacher obtains tau-a 0.591
and top-1 0.842 against exact remaining-step order.  Replacing the goal
offline by a deterministic necessary-only trace improves these to 0.623 and
0.870 (especially on demonstrations containing distractors), which diagnoses
distractor-contaminated terminal goals; this construction uses symbolic
ancestry and is audit-only.  The deployable trace-only alternative, prompt
plus the observed final answer sentence, fails badly (tau-a -0.084, top-1
0.454) because it is off-distribution relative to complete rollout states, so
it was not allocated a training run. The matched greedy horizon screen is
complete for H={1,2,4,8,16}; H=16 reaches 0.655/0.895 with teacher top-1
0.790, again selecting H=2 among width-one teachers.
An additional fully non-symbolic audit used four independent terminal traces
generated by a uniformly random feasible policy. Neither their mean goal nor
nearest-set distance is viable: on 100 anchors, teacher/oracle tau-a and top-1
drop from 0.679/0.89 for the demonstrated goal to 0.285/0.64 (mean) and
0.339/0.65 (set). The state encoder is trajectory-specific, so no multi-goal
training run was allocated. Alternative-goal diagnostics are skipped for
greedy checkpoints, which do not retain offline shooting leaves.

The faithful deterministic LDAD stability matrix is complete. It is a clean
3 target modes (EMA / online stop-gradient /
fully online gradients) x 3 anti-collapse settings (none / VICReg / SIGReg) x
raw-token LDAD off/on factorial, using 30k fresh examples per epoch for 10
epochs. Only one-step JEPA prediction, the selected regularizer, and optional
observed-token decoding are present; GAR, value supervision, outcome anchors,
rollout, hierarchy, and the legacy hybrid operation target are absent. The
base config is `disc_observed_ldad_factorial`, the runner is
`scripts/run_observed_ldad_cell.sh`, and the live table is
`runs/observed_ldad_factorial.md`. LDAD weight is deliberately 1.0 here to
give the hypothesis that observed-action reconstruction alone prevents
collapse its strongest controlled test; the separate recipe weight screen
selects the performance-oriented coefficient.
The first EMA rows already localize LDAD's role. Without VICReg/SIGReg,
raw-token LDAD raises counterfactual matching 0.432 -> 0.959 and RSA
0.300 -> 0.737 (token accuracy 0.910, exact phrase 0.584). With EMA+VICReg,
matching is already 0.998 without LDAD and remains 0.992 with it; EMA+SIGReg
without LDAD is 0.990. The online-stop-gradient/no-regularizer pair is also
decisive: LDAD raises state std/rank from 0.102/2.95 to 0.343/101.8 and
matching from 0.299 to 0.963. It therefore prevents severe collapse without
EMA in this controlled setting, although VICReg/SIGReg retain healthier unit
state scale. The fully online/no-regularizer pair is also decisive: LDAD
changes state std/matching from approximately 0.000/0.322 to 0.145/0.988
(rank 90.3), so it prevents complete collapse and grounds transitions without
EMA or stop-gradient. It still does not yield unit-scale geometry or guarantee
a calibrated planning energy. The final fully-online VICReg pair shows that
healthy distributional statistics do not imply transition grounding: VICReg
without LDAD has state std/rank 1.016/234.5 but matching 0.334; adding LDAD
gives 1.007/219.6 and matching 0.980 (RSA 0.558 -> 0.909).

The core one-step mechanism above is faithful to Delta-JEPA's displacement
input and externally observed raw-action target, but is a discrete-text
analogue rather than an architecture-identical visual-control replication.
The paper's multi-step extension is now implemented separately:
`MultiStepObservedActionDecoder` reconstructs H ordered intent phrases from
`s_{t+H}-s_t` alone with displacement-conditioned Transformer blocks. Three
fully online two-objective controls are complete:
`deltajepa_text_{noldad,h1,h4}`. They use unnormalized latent MSE, raw-token
action reconstruction when enabled, no EMA, no stop-gradient, and no
VICReg/SIGReg. This comparison is independent of the GAR horizon sweep.
Prediction-only collapses to state std
0.00028 and matching 0.307 (chance 0.288); adjacent LDAD gives state std/rank
0.117/89.5, matching/RSA 0.975/0.775, and token/exact recovery 0.909/0.584.
The H=4 sequence decoder gives state std/rank 0.080/140.1, matching/RSA
0.994/0.804, and token/exact-sequence recovery 0.855/0.406. The stricter exact
metric requires all four ordered phrases to be correct. Thus the paper-style
multi-action objective also prevents magnitude collapse and identifies
counterfactual transitions in text.

The full-recipe extreme online control is complete and negative for the claim
that LDAD alone replaces all stabilization. With no EMA, no stop-gradient and
no VICReg, adding faithful LDAD changes strict/slack-2 planning from
0.565/0.795 to 0.440/0.785. It improves transition matching 0.456 -> 0.701
and state rank 79.1 -> 107.5, but degrades learned-value tau 0.793 -> 0.705
and top-1 0.870 -> 0.799. Thus action recovery and selection-energy quality
are separable. The exact no-EMA-only result is reported above; the separate
no-stop-gradient-with-VICReg control remains diagnostic rather than a shear
delta.

Official iGSM now supports the same geometry-greedy GAR interaction: reference
environment execution, reference-rendered outcome text, and tuple-valued
official action identifiers all pass through the shared collator/model path.
The first clean transfer pilot, `real_latent_goal_h2_r1`, is now training on
an idle L40. It exactly transfers the current reduced direct-predictor model:
one-step latent prediction, on-trajectory outcome prediction and its internal
predicted-outcome rollout, VICReg, EMA, and H=2/K=2 multi-step latent-goal
preference distillation. It contains no symbolic quality label, LDAD,
monotonicity, hierarchy, explicit open-loop latent loss, scalar value
regression, counterfactual transition auxiliary, or residual skip. This is a
transfer pilot rather than a frozen official recipe; any stylized add-back
accepted by the matched selection gate must subsequently be tested here.
A faithful deterministic LDAD pilot on the strong symbolic-preference
reference reaches **0.535 strict / 0.805 slack-2**, versus the historical
`real_rank` 0.410/0.735; its repaired official counterfactual audit gives
transition matching 0.956 and value-order tau 0.796. This is not an LDAD
effect estimate because the new
run also uses fresh generated data every epoch whereas the historical run
reused one corpus. The exact fresh-data no-LDAD counterpart
`real_rank_fresh_noldad` is now complete. Without LDAD, the otherwise matched
fresh-data model reaches **0.565 strict / 0.805 slack-2**, transition matching
0.889, value tau 0.791, and RSA 0.683. With LDAD the corresponding values are
0.535/0.805, 0.956, 0.796, and 0.786. Faithful displacement reconstruction
therefore improves transition identity and representational alignment in this
official symbolic-preference configuration, but not planning success; under
the shear rule it is removed from this configuration. The matched 200-episode
audit reinforces the decision: without/with LDAD, clean-history useful-action
top-1 is 0.848/0.837 and after-error top-1 is 0.653/0.623. In the 10--20
distractor-variable stratum, failure is 0.632 without LDAD versus 0.789 with
it. The healthiest no-EMA sentence-stream setting
(online stop-gradient + SIGReg, no posterior-code reconstruction) is active
on official iGSM. This action-free auxiliary is not faithful observed-action
LDAD because its target is an inferred posterior code rather than an external
action.
Its first launch accidentally used SIGReg weight 1.0 instead of the
factorial-selected 0.01; this was caught after epoch 1, stopped, cleaned, and
restarted at 0.01. The semantic variational probe is chained after training.
`scripts/probe_variational.py` audits semantic action content and
matched-vs-shuffled action sensitivity rather than treating variance/rank as
semantic evidence.

The full 200-episode closed-loop failure audit for
`real_rank_faithful_ldad` is complete. Overall strict success is 0.535.
Competitive useful-action top-1 is 0.837 on clean histories and 0.623 after a
previous distractor; only 2.3% of competitive decisions are near ties within
0.02. Failure rises from 0.225 for problems with 0--4 distractor variables to
0.589 for 5--9 and 0.789 for 10--20. Necessary-chain length is not monotone in
the same way (failure 0.412/0.489/0.500 for 1--4/5--8/9--15 necessary
actions). The official pilot is therefore primarily distractor-load and
off-history-recovery limited. Because this model uses symbolic preference
training, it remains a diagnostic boundary rather than the target method. Its
fresh-data no-LDAD counterpart is complete and supplies the matched LDAD
effect reported above; the clean non-symbolic transfer pilot is active.

`DiscourseVJEPA` now supplies a controlled probabilistic-state factorial with
the same discourse encoder and transition architecture across inferred latent
actions, pooled observed intents, and concatenated token-bottleneck intents.
Pooled/token observed-action cells each run with and without raw-token LDAD;
all five use online stop-gradient + SIGReg weight 0.01 and are complete.
Official-iGSM token-LM, sentence-LM, and
sentence-LM+latent-target training/planning now use the faithful vocabulary,
reference renderer, and reference feasible-action interface. Their first
matched seed is now training on three idle L40s because these baselines do not
depend on the pending JEPA add-back decision. The token model will be evaluated
by intent likelihood; the sentence model by intent reconstruction likelihood;
and the sentence+latent model by both reconstruction likelihood and latent
distance. Additional seeds remain gated on the frozen comparison protocol.

All five discourse-variational cells now have diagnostics in
`runs/discourse_variational.md`. The inferred posterior code is not a usable
action proposal: shuffling it worsens prediction 7.24x and it decodes the
next value at 0.992, but the prior retrieves the matching posterior at only
0.0015 top-1 (MRR 0.011) and decodes value at 0.184. This is outcome-residual
inference, not action discovery. Pooled/token observed actions give calibrated
Gaussian predictions (standardized residual squared about 0.99); pooled raw
LDAD raises state rank 88.1 to 107.1 and displacement-value decodability
0.855 to 0.992, while action sensitivity changes 1.395 to 1.305. With the
token-concatenated encoder, LDAD improves matched prediction error
0.517 -> 0.453, action sensitivity 1.151 -> 1.269, action rank
4.60 -> 7.61, and displacement operation decoding 0.912 -> 0.992. Hence
the auxiliary's effect depends on action representation.
The completed bottleneck sweep shows that this is an order effect rather than
extra action-encoder capacity. Keeping only 2 dimensions per ordered token
(6.79M backbone parameters, versus 6.81M at width 8 and 6.85M for pooling)
gives essentially the same
LDAD-on matched error/sensitivity/token recovery: 0.450/1.259/0.886, versus
0.453/1.269/0.886 at width 8. Width 4 is likewise indistinguishable at
0.456/1.257/0.887. Use width 2 as the minimal ordered-token architecture in
subsequent bottleneck studies; these cells do not establish that it is better
than pooling overall.

The strongest action-free sentence-stream pair now has both semantic probes
and a same-state counterfactual-coverage audit. With online stop-gradient +
SIGReg on stylized data, posterior-code displacement reconstruction improves
matched L1 0.330 -> 0.256 and distinct posterior candidate matching
0.285 -> 0.645. Across 199 states, 64 prior samples cover 0.052/0.278 of
geometrically distinct true-outcome regions without/with the auxiliary and
0.155/0.494 of posterior modes. However, only 0.004/0.016 of candidates get a
prior prediction as accurate as the correctly grounded posterior. This is
better posterior mode separation and broad support, not accurate controllable
action discovery.
The controlled prior-alignment diagnostic is complete. Keeping action-KL at
0.1 while removing the four-free-nats allowance makes posterior and prior
nearly identical (cosine 0.996), but by collapsing the action modes:
posterior distinct matching falls 0.285 -> 0.006, prior distinct outcome
coverage falls 0.052 -> 0.010, and posterior-mode separation is only
9.2e-5. Posterior/prior prediction L1 both become 0.518 and shuffled-action
sensitivity is 1.000. State scale/rank remain superficially healthy at
0.991/145.0, showing again that global state statistics cannot diagnose
same-state controllability. Do not launch a stronger-KL cell: zero free nats
already moves in the harmful direction. The paired table is in
`runs/action_free_transfer.md`.
The controlled action-free prior-architecture diagnostic is complete.
`MixtureVariationalAction` retains the same
transition-informed Gaussian posterior but replaces the single diagonal-
Gaussian plan-time prior with four context-conditioned Gaussian components;
its differentiable objective is the log-sum-exp mixture-KL upper bound. The
four-component run keeps online stop-gradient, SIGReg, action-KL 0.1, four
free nats, and no posterior-code reconstruction. This is the only causal
change. It does not help: prior distinct coverage is 0.050 versus 0.052 for
one Gaussian, precise coverage 0.005 versus 0.004, posterior distinct matching
falls 0.285 -> 0.230, and posterior-mode coverage falls 0.155 -> 0.093.
Effective component use is only 1.36/4, mean maximum component probability is
0.924, and aggregate weights are [0.017, 0.021, 0.924, 0.038]. The mixture
has collapsed to one component. Do not combine it with posterior-code
reconstruction. Learning multiple prior modes likely requires observing a
same-state set of counterfactual outcomes, which is a distinct environment-
interaction experiment rather than a capacity-only follow-up.

That experiment is now implemented without exposing intents or quality
annotations. `SentenceStreamVJEPA(counterfactual_set=true)` encodes every
feasible rendered alternative next sentence, trains the posterior-conditioned
transition on every outcome, and fits the plan-time prior to every resulting
posterior mode. Three official configurations form the controlled completion:
single-Gaussian prior plus outcome set, four-component prior without the set,
and four-component prior plus outcome set. `data.n_alt=64` exceeds the domain
branching factor, so the set is exhaustive. All configurations compose, both
prior types pass forward/backward tests, and an official faithful-renderer
smoke test gives finite losses and gradients to both transition and prior.
The three missing cells are active: Gaussian+set on gruenau11 GPU 0,
mixture-without-set on gruenau9 GPU 1, and mixture+set on gruenau11 GPU 3.
The existing single-Gaussian/no-set model is the fourth factorial cell. Each
detached chain trains, runs `probe_variational.py`, runs the 64-sample
same-state audit, refreshes `runs/action_free_transfer.md`, marks `DONE`, and
removes only its redundant `last.pt`. Counterfactual selection uses an independent
per-example RNG, and the official renderer's global Python/NumPy RNG state is
restored after rendering alternatives. Regression tests therefore verify that
turning the outcome set on leaves every factual trajectory and geometric-
teacher field byte-for-byte unchanged on both domains; the set/no-set cells
are exactly paired rather than merely distribution matched.
The first exhaustive-set batch exposed the CUDA SDPA launch-grid limit when
thousands of independent histories were flattened into one attention call.
`_encode_stream_batches` now encodes at most 256 histories per call; this is
mathematically exact because histories never attend across examples. A
regression test checks the 256/256/remainder split, the full suite is green,
the two empty failed run directories were removed, and both set cells were
relaunched on their original GPUs.

The official same-state audit shows why global variance/rank and demonstrated-
transition prediction are insufficient. EMA+VICReg has state std/rank
0.988/151.2 and posterior/prior L1 0.0055/0.0063, but alternative true outcomes
at the same state have separation only 0.005; posterior distinct matching and
prior distinct coverage are both zero. Online stop-gradient+SIGReg has lower
global rank 40.9 but true-outcome separation 0.989, posterior distinct matching
0.753, and broad prior-region coverage 0.598. Only 0.073 of candidates are
covered at posterior-level accuracy, and necessary actions receive 0.376 of
assignments versus a 0.332 candidate share. The exact official posterior-code-
reconstruction counterpart is complete. It improves prior-mean/best-of-8 L1
from 0.966/0.601 to 0.589/0.449 and broad distinct-outcome coverage from 0.598
to 0.675. However, posterior L1 worsens from 0.006 to 0.049, true-outcome
separation shrinks from 0.989 to 0.743, and posterior-accuracy coverage falls
from 0.073 to 0.029. The auxiliary therefore helps broad pre-transition
support but not precise action-free control. Full definitions and rows:
`runs/action_free_transfer.md`.

A separate faithful observed-action *variational* LDAD factorial is now ready
and must not be confused with either matrix above. Its base is
`dvjepa_observed_ldad_factorial`; `scripts/run_observed_vjepa_ldad_cell.sh`
crosses probabilistic-state target mode (EMA / online stop-gradient / fully
online gradients), regularizer (none / VICReg / SIGReg), and raw-token LDAD
off/on for 18 cells. It uses pooled observed intents, 30k fresh examples per
epoch, 10 epochs, and LDAD weight 1.0. The live table is
`runs/observed_vjepa_ldad_factorial.md`; the three target-gradient paths and
observed decoder pass dedicated tests. All three no-regularizer target-mode
pairs are complete.
LDAD off/on gives state std 0.00075/0.00454 and matched-vs-shuffled action
sensitivity 1.004/1.152. The LDAD-on decoder nevertheless reaches 0.898 token
and 0.582 exact-phrase recovery, while standardized residual squared remains
apparently calibrated at 1.012. Thus, unlike the deterministic JEPA, faithful
LDAD alone does not prevent probabilistic-state magnitude collapse: the token
decoder can amplify tiny displacement directions and learned variance can
calibrate a collapsed mean. The fully-online/no-regularizer and EMA/no-
regularizer pairs confirm the same distinction. Fully online LDAD off/on
gives state std 0.00034/0.00305 and action sensitivity 1.000/1.035, despite
0.900 token and 0.583 exact-phrase recovery. EMA improves scale, but its
off/on pair gives state std 0.062/0.286, state rank 76.4/52.3, and sensitivity
1.182/1.455; faithful LDAD improves action dependence while reducing
effective rank. The full 18-cell matrix is now complete. The strongest
balanced pooled-action cell is EMA+SIGReg+LDAD: state std/rank 0.970/80.2,
matched L1 0.062, shuffled/matched error 2.293, and calibrated standardized
residual squared 0.983. EMA+VICReg+LDAD gives lower matched error 0.018 and
higher rank 90.1 but weaker action sensitivity 1.647. The architecture
follow-up is complete. At EMA+SIGReg, raw LDAD with the pooled residual
predictor gives matched L1/sensitivity/state rank 0.062/2.293/80.2. The
two-dimensions-per-token ordered bottleneck gives 0.102/2.086/89.3 and does
not dominate pooling. The pooled direct predictor gives
0.245/2.990/171.8 with LDAD, versus 0.501/1.728/161.4 without: direct
prediction preserves much richer state geometry and the strongest action
dependence, while the residual parameterization fits the next-state mean more
accurately. The EMA+VICReg direct pair confirms the rank effect: LDAD off/on
has state rank 221.1/213.6 versus 97.2/90.1 for the residual pair. Within the
direct model LDAD improves matched L1 0.469 -> 0.320 and sensitivity
1.415 -> 1.835. SIGReg gives the strongest action dependence; VICReg the
richest state geometry. Full table: `runs/variational_architecture.md`. Do
not transfer the deterministic anti-collapse conclusion to the probabilistic
model.

An exact official-iGSM transfer of pooled EMA+SIGReg with raw LDAD off/on is
complete as `dvjepa_faithful_pooled_ema_sigreg{,_ldad}`. Faithful raw-action
reconstruction changes matched L1 **0.463 -> 0.067** and shuffled/matched
action sensitivity **1.105 -> 1.462** while state std remains 0.981/0.979;
state rank decreases 119.7 -> 102.9. The LDAD-on decoder reaches token/exact
recovery 0.942/0.730 and standardized residual squared remains calibrated at
0.967 versus 0.991. This is the first exact cross-domain probabilistic raw-
action LDAD effect. Official outcome
sentences exceed the stylized 48-token chunk limit, so these configs use the
already-established official limit of 96; both sides passed a long-example
forward smoke before launch. The live table is
`runs/variational_transfer.md`.

The official ordered-token transfer is also complete. Relative to the pooled
residual action encoder, the two-dimensions-per-token bottleneck preserves
state std/rank but lowers action-code rank from 14.9/14.5 to 6.6/7.8 without/
with LDAD. LDAD-on ordered versus pooled conditioning gives matched L1
0.081 versus 0.067 and shuffled/matched sensitivity 1.222 versus 1.462.
Exact-phrase reconstruction improves slightly from 0.730 to 0.775, but this
does not produce a better transition control variable. Mean pooling is
therefore retained for observed-action probabilistic dynamics.

The exact official direct-predictor pair is also complete. Removing the
residual state skip increases LDAD-off state rank from 119.7 to 185.3 but
makes the conditional mean harder to fit (matched L1 0.463 -> 0.719) and less
action-sensitive (1.105 -> 1.097). Within this direct architecture, faithful
LDAD lowers matched L1 **0.719 -> 0.266**, raises shuffled/matched sensitivity
**1.097 -> 1.457**, and retains healthy state std/rank 0.987/163.5. Token and
exact-phrase recovery reach 0.978/0.893 and standardized residual squared is
0.999. Thus the cross-domain LDAD effect survives removal of the residual
skip; residual prediction favors fidelity, while direct prediction favors
richer global state geometry.
The official EMA+VICReg direct pair is now complete as well. Faithful LDAD
changes matched L1 **0.612 -> 0.199**, sensitivity **1.104 -> 1.307**, and
state rank 243.7 -> 197.3; token/exact recovery is 0.979/0.901 and residual
squared is 0.986. Against official SIGReg direct LDAD, VICReg gives lower
error (0.199 vs 0.266) and higher state rank (197.3 vs 163.5), while SIGReg
gives stronger action sensitivity (1.457 vs 1.307). Neither regularizer
dominates every diagnostic.
The stylized fully-online-gradient + SIGReg direct-predictor pair is complete.
The matched residual cells have healthy marginal scale but only rank
14.1/17.5 and almost no action dependence (shuffle ratios 1.054/1.138).
Switching to direct prediction makes the collapse still more explicit: raw
LDAD off/on gives state rank 8.5/6.3 and action-code rank 1.27/1.42. The large
shuffle ratios 51.4/48.3 are denominator artifacts because matched L1 is only
0.0037/0.0049; nearly all learned action dependence occupies one direction.
Therefore neither direct prediction nor faithful LDAD replaces the EMA target
in this probabilistic setting. `runs/variational_architecture.md` reports the
complete matched table.

The matched observed-action open-loop uncertainty audit is complete for
stylized and official iGSM with pooled EMA+SIGReg residual predictors. It
fixes the true intent sequence and recursively propagates 32 Gaussian samples
over 1,000 validation examples. Raw-action LDAD substantially improves the
conditional mean: stylized teacher-forced normalized L1 at horizons 1/4 is
0.396/0.431 without LDAD versus 0.026/0.095 with it; official iGSM gives
0.402/0.472 versus 0.051/0.071. Raw LDAD-on spread is under-dispersed at long
horizons: official open-loop L1 grows from 0.052 at H=1 to 0.279 at H=8 while
two-standard-deviation coverage falls from 0.949 to 0.779. The no-LDAD model
is inaccurate but increasingly over-dispersed. A held-out scalar calibration
audit fits one standard-deviation multiplier per horizon on the first 500
trajectories and evaluates the second 500. At official H=8, LDAD-on needs
1.684x spread, changing residual squared 2.827 -> 0.997 and coverage
0.777 -> 0.952; no-LDAD needs 0.612x, changing coverage 0.998 -> 0.953.
Across all domain/horizon cells, calibrated residual squared is 0.969--1.079
and coverage is 0.946--0.964. Faithful LDAD therefore grounds conditional
means, while a simple horizon-dependent scalar corrects marginal recursive
spread; accumulated mean bias remains. The schedule is also frozen and
applied without refitting to 1,000 disjoint test trajectories per model. At
official H=8, test residual squared changes 0.375 -> 1.003 without LDAD and
2.840 -> 1.002 with LDAD; two-sigma coverage changes 0.998 -> 0.953 and
0.776 -> 0.952. Across H=1/2/4/8, transferred test residual squared is
0.964--1.084 and coverage is 0.943--0.962. This establishes cross-split
spread calibration rather than a within-validation fit artifact. Full counts are in
`runs/variational_rollout.md`; regenerate with
`scripts/report_variational_rollout.py`.

Baseline-interface correction: the historical LM planners enumerate the same
feasible actions but score rendered next-step sentences containing the true
computed candidate values. Keep those as strong **outcome-candidate**
diagnostics, not parity baselines. New `target_kind=intent` token/sentence LM
variants train on interleaved selected intent and observed-outcome histories,
apply CE (and optional next-latent regression) only to intent phrases, and
rank the exact outcome-free phrases seen by JEPA. The matched 9M token policy
is complete over three seeds at **0.827 ± 0.003 strict / 0.978 ± 0.003
slack-2**, so the final non-symbolic
latent model must exceed a substantially stronger baseline than the old
outcome-candidate number suggested. The sentence policy is complete over
three seeds at **0.690 ± 0.053 strict / 0.903 ± 0.039 slack-2**, below the
token policy. The
sentence-policy-plus-latent model is now complete over three seeds:
intent-likelihood selection reaches **0.817 ± 0.034 / 0.953 ± 0.010**, while
latent-distance selection reaches **0.840 ± 0.017 / 0.965 ± 0.013**. Faithful
variants are configured but wait for the stylized recipe.
The live exact-interface comparison is `runs/matched_baselines.md`; regenerate
it with
`scripts/report_matched_baselines.py --domain stylized --split val --jepa <selected-run>`
whenever a new combined model finishes. After the
guarded final evaluation, replace `val` by `test`; the reporter reads only
`*_test.json` artifacts and writes `runs/matched_baselines_test.md`.

Paper protocol correction: configs now define independent train/validation/
test generator seeds (1/2/3). Recipe selection and shearing use validation;
only the frozen final models and baselines are evaluated on `split=test`.
Regression tests now assert val/test seed and generated-problem disjointness
for both stylized and official faithful generators. Planner outputs carry a
`_test` suffix, so test artifacts cannot overwrite validation results.
`scripts/eval_final_test.sh` additionally refuses to open the test split
unless `FINAL_TEST_CONFIRM=recipe-frozen` is set; it supports latent, token,
sentence, and sentence-plus-latent policies under the same strict/slack-2
protocol.
Historical headline numbers are validation numbers until rerun under this
protocol.

Artifact cleanup at 20:15 identified 159 completed/evaluated runs with both a
retained best checkpoint and a redundant final checkpoint (10.1 GB total).
Only those redundant `last.pt` files and Python/pytest caches were removed;
active jobs, every `best.pt`, metrics, probes, and planning/audit artifacts
were retained. `runs/` fell from about 24 GB to 15 GB. At 22:42 three more
completed diagnostic `last.pt` files and regenerated repository caches were
removed. At 05:50, after all corresponding CSV/JSON reports were verified,
best/final checkpoints from completed rejected GAR, counterfactual, LDAD
factorial, selector-margin, and reduced-model add-back cells were removed.
Selected references, active jobs, final baselines, and checkpoints needed by
pending audits were preserved; `runs/` fell from 18 GB to 12 GB.

## 6. THE ACTIVE RESEARCH QUESTION (read this before doing anything)

The user's directive: **eliminate symbolic supervision from the recipe**,
because any symbolic ingredient (a) weakens the claim that JEPA itself
solved the task and (b) blocks transfer to real language. Supervision
taxonomy now used everywhere:
(a) trace-only (text + geometry — transfers to real language),
(b) environment interaction with GEOMETRIC labels (transfers if you can
    sample counterfactual continuations),
(c) symbolic annotations (counts/flags — do NOT transfer).

The completed one-step **geometric-advantage ranking (GAR)** test,
`GeoAdvantageRank` in `src/textjepa/objectives/ranking.py` + `_geo_rank`
in `models/discourse_jepa.py` + `geo_rank_k` in the dataset: at ONE anchor
step per trace, the env executes K alternative actions (returning outcome
TEXT only); the EMA teacher encodes those true next states; their LN-L1
distance to the EMA terminal goal orders `V(F(s,a_i))` via a margin loss
(label_gap 0.02 filters geometric ties). The GAR objective is tier (b). The
old `disc_georank*` pilots retained the operation-type displacement target,
so they were preference-label-free but not action-annotation-free. The new
clean greedy-GAR runs set that legacy objective to zero; their LDAD-on cells
decode only externally observed intent tokens. All variants retain the common
environment feasible-action/text interface. The old one-step pilot reached
only `0.440/0.845`; additional
counterfactual transition data was much stronger (`0.735/0.960` at K=2), but
that control still uses scalar progress and operation labels. The active clean
combined models remove counts, relevance flags, symbolic order, and operation
classes while testing 2/4/8/16-step geometric rollout advantages.

## 7. Next and further steps

1. **Select and shear the clean recipe to a fixed point.** Use strict-budget
   accuracy as primary and slack-2 as secondary. Remove every component whose
   one-seed effect is within 0.02, combine all such removals, rerun, then
   repeat one-component removals around the smaller model. Differences in the
   0.02--0.05 ambiguity band get a second seed before a decision. After two
   matched seeds, retain a component when its mean worst-case loss exceeds
   0.02; otherwise remove it. This rule is implemented by
   `scripts/shear_decision.py`, so replication cannot recurse indefinitely. Only the
   fixed-point recipe and its final one-component ablations receive three
   seeds. Monotonicity has no protected status and has already been removed.
2. **Diagnose the remaining action-free bottleneck.** Same-state
   posterior/prior coverage, prior-alignment, prior-capacity, both official
   posterior-code-reconstruction conditions, and observed-action open-loop
   uncertainty audits are complete. The reconstruction auxiliary improves
   broad prior support but reduces precise coverage; do not repeat seeds or
   claim action discovery from variance/rank or broad coverage alone. A new
   follow-up must change the information available to the prior, for example
   same-state counterfactual outcome-set supervision or explicit goal
   conditioning, rather than merely enlarge the prior.
3. **Use the completed latent-displacement factorial to select follow-ups.**
   Transition matching, likelihood calibration, raw-action reconstruction,
   semantic probes, and frozen validation-to-test spread calibration are
   complete. The validation-fitted horizon schedule transfers on both domains;
   accumulated mean bias remains. If an action-free planner is attempted,
   compare prior sampling,
   nearest-imagined-state matching, and a decoder only as an analysis probe.
4. **Fresh-data matched reruns.** The new epoch-offset sampler makes every
   epoch a deterministic disjoint generated corpus even with persistent
   workers. Re-run the main data-scale claims with matched optimizer steps;
   historical 10k/30k/100k artifacts confound corpus size and update count.
5. **Per-sentence edit encoding (both sides)** — lifts the edit audit cap
   (encoder-side, ~0.50); `AttnEditPredictor` exists, the encoder is next.
6. Paper assembly: refresh the self-contained 31-slide NeurIPS-style deck as
   the selector add-backs and official transfer resolve. Preserve the current
   structure: task/protocol, minimal method, main comparison, fixed-point
   ablation table, official-iGSM transfer, variational exploration, and
   limitations. Artifact names stay in the appendix; scientific names remain
   in the main text.

## 8. Gotchas that will bite you if you don't know them

- **Faithful iGSM**: seeding MUST use the md5 helper (`_fix_seed`) —
  builtin `hash()` is process-salted. The RNG pseudo-node (layer −1) is
  excluded from params/deps. Solutions name params WITHOUT the "each"
  prefix. `max_chunk_len: 96` required (sentences up to ~61 tokens).
  Vocab is cached at `configs/faithful_vocab.txt` (delete to rebuild).
- **Sentence lengths**: stylized ≤48 tokens; faithful needs 96; hard-domain
  LM token streams need `model.max_len=768`.
- `eval_run.sh` uses `.venv2`. Anything importing the vendored iGSM needs
  `transformers` in that venv.
- `scripts/analyze_failures.py` now auto-selects `FaithfulPlanner` for
  `igsm_real` checkpoints and records the same post-hoc decision margins and
  depth/distractor strata as on stylized iGSM. Its official one-episode CPU
  smoke and the unchanged stylized path both pass.
- The planner's `EpisodeResult`/plan JSONs: keys are
  `latent_planner[_oracle_goal]`, `lm_{outcome,intent}_policy`,
  `sentlm_{outcome,intent}_*`, `var_planner`.
  `scripts/report.py` aggregates everything into `runs/report.md`.
- Lookahead 1 already enumerates all currently feasible actions. Lookahead
  greater than one is not model-only: it uses the reference graph/environment
  to enumerate future feasible actions and detect terminal sequences. The
  planner now rejects it unless `allow_oracle_future_actions=true`, and output
  filenames carry `_oracle_actions`. Never put these diagnostics in the main
  fair-comparison table.
- Old checkpoints load with `strict=False` (new heads init randomly and a
  note is printed); config keys added later are read with `.get(...,
  default)` in `build_dataset`/`load_run` for backward compat.
- Deck compiles need two pdflatex passes; figures regenerate from run
  artifacts automatically (`make_report_figs.py` skips missing files).

## 9. Documentation map

- `RESULTS.md` — findings 1–21 + probe section + tables.
- `runs/current_results_analysis.md` — compact decision-oriented tables for
  the current deterministic and probabilistic experiments, including pending
  fixed-point cells.
- `runs/selector_decision.md` — predeclared complexity-aware H/B planning
  decision, generated by `scripts/selector_decision.py`.
- `runs/action_grounding.md` — aligned-versus-permuted observed-action
  falsifier, generated by `scripts/report_action_grounding.py`.
- `reports/discourse_jepa_neurips.pdf` (currently 31 slides) — self-contained
  paper-style task, protocol, method, objective selection, faithful transfer,
  variational studies, limitations, and artifact appendix.
- `reports/edit_jepa.pdf` (9 slides) — edit track.
- `runs/report.md` — auto-generated master table (energy × slack × look).
- `tests/` — 51 tests; run before committing model/objective changes.

## 10. Current priority override — hierarchy repair (2026-07-14)

The user paused every hard-text/variational experiment. Do not launch those
or resume recipe shearing. The only active goal is to make hierarchical
planning work in the observed intent-phrase playground, where failures can be
isolated quickly. The exact ledger is
`research/intent_phrase/waves/08_hierarchical_planning.md`; the staged gate is
`research/intent_phrase/BACKLOG.md`.

The HWM-scale CEM implementation uses 1200 candidates, 20 refits, 10 elites,
variance EMA .9, a state-conditioned macro prior, and explicit convergence
traces. Controls show the optimizer converges but exploits learned dynamics;
CEM budget is not the present bottleneck. A true K-step waypoint plus
oracle-feasible K-step lower search reaches .84 strict, equal to the flat
value planner, while learned macro planning reaches only .07-.15. Valid
counterfactual macro spans have trained the state-value, action-value/ranking,
and support heads to .822, .831, and 1.000 pair accuracy, respectively, but
the high waypoint remains inconsistent with the lower rollout and continuous
codes leave the valid macro manifold.

The completed `intent_hreach_d32_w{025,1,4,16}` sweep shows that moving the
high prediction toward the frozen low endpoint reduces high-to-low L1 from
.386 to .228, but does not improve the oracle-lower control (.03-.08) and
hurts true-state fidelity. `intent_hmultisupport_d32` improves .25-sigma
support discrimination .83 -> .924 and moves CEM codes closer to valid spans
(2.98 -> 2.21 RMS), but gives .07 oracle-lower/.08 deployed success. Exact
KNN support/projection also gives only .05-.07. These approaches are rejected.

Endpoint-only or dense lower-dynamics supervision reduces recursive K-step
error .580 -> about .485; the combined endpoint model raises exact
high-waypoint-to-low-span retrieval .550 -> .725, or .805 when high-to-low
reachability is also added. Listwise top-1 training raises terminal-optimal
selection .915 -> .950 but does not fix behavior.

The decisive 400-anchor stratified retrieval audit shows that the learned
macro begins with a necessary action .99/.96 at true remaining distance 3/4
and .83--1.00 beyond four. It fails only at distance 1--2 because the planner
requires a full K=3 span, forcing distractor padding near termination. A
newly distilled exact-distance `V(s)` (validation MAE .49) now terminates the
macro option and switches to one-step value control near the goal. This raises
oracle-valid discrete hierarchy from .08 to .81 at threshold 2.75--3.0 while
still using macro decisions 40--48% of the time; threshold 3.25 reaches .82
with 34% macro use. Flat value is .84. Fully learned HWM CEM rises from .08 to
.65 at threshold 3.0 with 43% macro use. Threshold 4 reaches .76 but uses the
hierarchy only 14%, so it is not the preferred operating point. Final-mean CEM
beats best-sample return (.65/.63), and nearest-span projection hurts (.59).

`intent_hvalue_full` is negative: dense open-loop + endpoint repair gives .79
oracle-valid/.61 CEM and does not beat the simpler .82/.65 checkpoint. True
waypoint + oracle-feasible lower search reaches .84, whereas true waypoint +
unrestricted learned lower search reaches only .63. Removing the learned
future-action support penalty actually raises that lower control to .70;
support weights and hard thresholds do not solve it. Adding a lower terminal
value reaches .65-.66, but a zero-subgoal ablation gives the same result, so
that lever is a disguised value-only controller and is rejected.

The decisive repair is a goal-conditioned closed-loop lower option head
`Q_low(s,a,s_subgoal)`. A macro subgoal is set-valued under commuting actions,
so its correct multi-positive target is any currently feasible action
contained in the target span, not the arbitrary recorded first action. The
head trains against both true and predicted macro subgoals and never
enumerates symbolic future availability.

`intent_hgoalpolicy_set_a8_w1` reaches .954/.881 top-1 on true/predicted
subgoals. Its preserved epoch-2 checkpoint gives .82 discrete-all/.80 CEM on
the 100-episode screen. Report the larger result instead: on three 500-problem
test seeds, HWM CEM at termination threshold 3.25 gives .766/.762/.776 (mean
.768) with .296/.302/.297 macro-decision rate, versus matched all-flat
.798/.800/.814 (mean .804). The working fully learned hierarchy is thus 3.6
points below flat while genuinely using macros for 30% of decisions; the
previous fully learned result was .65-.66. Threshold 2.5 is the utilization
point (.754 test at 51% macro use), not the primary performance setting.

The exact-order control gives only .69 CEM/.60 true-waypoint and is rejected.
The final multi-positive checkpoint improves the auxiliary probe to
.972/.966, but gives .746 on the 500-val CEM control at threshold 3.25; at
threshold 3.75 it only ties .770 while using macros .197 versus .286 for
epoch 2. The Pareto-selected artifact is
`runs/intent_hgoalpolicy_set_a8_w1/selected_planning.pt` (epoch 2), not
`best.pt`. Do not select on auxiliary loss. Full tests pass (59 tests). Do not
launch hard-text or variational work. Always report macro-decision rate so a
mostly-flat controller is not mislabeled hierarchy.

The current suite is 58 tests; invoke it with
`.venv/bin/python -m pytest tests -q` (plain system `pytest` lacks torch and
collects vendored third-party tests).
