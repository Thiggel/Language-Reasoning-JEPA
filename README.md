# TextJEPA

The three scientific subprojects have stable entry points under
[`projects/`](projects/README.md):

- [observed intent-phrase JEPA](projects/intent_phrase/README.md);
- [token-level causal iGSM JEPA](projects/token_igsm/README.md);
- [sequence-edit JEPA](projects/sequence_edit/README.md).

Shared implementation remains in `src/`, `configs/`, and `scripts/`; existing
run paths are preserved. Detailed experiment records are under
[`research/`](research/README.md). `HANDOFF.md` remains the operational history,
while `RESULTS.md` preserves chronological findings.

> **Continuing this project? Start with [HANDOFF.md](HANDOFF.md)** — current state, in-flight experiments, active research question, and every gotcha.

> **Results and protocol notes: see [RESULTS.md](RESULTS.md).** The current
> non-symbolic reduced JEPA averages **79.7%±0.8% strict / 96.3%±0.8%
> slack-2** over three seeds (random: 5.5%/40.5%; oracle: 100%/100%). It
> uses one-step intent-conditioned latent prediction, an outcome target,
> variance--covariance regularization, an EMA target, and H=2/K=2 multi-step
> latent-goal preference distillation (two alternative roots plus the
> demonstrated action). It still trails the matched token intent policy at
> 82.7%±0.3%/97.8%±0.3% over three seeds and is not frozen: H=4/B=8 and
> H=8/B=8 bounded-continuation students are the remaining selector gate.
> Faithful-LDAD and scalar-value add-backs and the softer preference margin
> are rejected. A three-seed
> symbolic-preference reference reaches **96.2%/99.7%** and is not the
> proposed method.

Joint-embedding predictive architectures for **language**: latent world models
over discourse with observed intent actions and latent-space planning. The
state and consequence models are reconstruction-free. Hierarchical and
multi-step objectives are optional controlled auxiliaries, not defining model
components. A controlled LDAD auxiliary may decode the externally observed
action text from a state displacement; it is removed after training and never
generates outcomes or enters planning.

The intent-phrase project also contains deterministic and probabilistic
variants, but the repository has three paper-level subprojects that must not
be conflated:

| track | world state | action | "generation" |
|---|---|---|---|
| **observed intent phrase** | compressed reasoning state | observed intent phrase ("derive X from A plus B") | choose the next currently feasible intent by learned latent energy |
| **token-level causal iGSM** | causal token-prefix state | token; compressed text span at higher levels | generate/plan tokens through multiscale latent dynamics |
| **sequence edit** | latent slots over a draft text buffer | delete/insert/replace intent | edit the buffer until correct |

## Why this setup

JEPA needs a *world* whose consequences are non-trivial to predict. Here the
world is an iGSM-style synthetic reasoning environment (a DAG of named
quantities with mod-p arithmetic): action phrases state **intent only**, so the
predictor must compute the *consequence* (e.g. the modular-arithmetic result
stated by the next step) purely in latent space. Because the data is synthetic
we have exact ground truth for every probe (values, resolved sets, remaining
steps, defect counts) and objective planning success criteria.

The core follows three papers:

- **I-JEPA and variational JEPA**: EMA target encoders, latent prediction,
  and probabilistic targets without reconstruction (arXiv:2301.08243;
  arXiv:2601.14354). The repository now also contains a full probabilistic
  sentence-stream model with latent, unobserved transitions.
- **Delta-JEPA** (arXiv:2606.31232): the paper decodes an observed raw action
  from `Δs = s_{t+1} − s_t`. The older TextJEPA hybrid instead decodes a
  symbolic operation class and an EMA intent embedding; it is useful
  displacement supervision, but not a faithful Delta-JEPA replication. The
  current observed-action LDAD reconstructs the complete externally observed
  intent-token sequence from `Δs` alone and is the discrete-language analogue
  used in the controlled stability factorial. The multi-step extension also
  reconstructs an ordered sequence of intent phrases from one long-horizon
  displacement, matching the paper's multi-action LDAD mechanism.
- **HWM, Hierarchical Planning with Latent World Models** (arXiv:2604.03208):
  macro-action encoder (CLS transformer over K action codes, deliberately tiny
  latent), high-level predictor in the *same* latent space, teacher-forcing +
  open-loop rollout losses, MPC-style planning with value/goal energies.

Stabilization: VICReg variance/covariance on online states + EMA targets.
Historical scalar remaining-step regression uses detached states by default;
the selected pairwise planning energy receives preference gradients, while
scalar regression is absent from the reduced model.

## Layout

```
configs/                 hydra config tree (data / model / objective / probe / plan)
src/textjepa/
  data/igsm/             problem DAGs, NL rendering, symbolic env, dataset
  data/edits/            corruption + repair trajectories, edit env, dataset
  models/                encoders, causal state models, deterministic and
                         variational predictors, EMA, shared dynamics core
  objectives/            latent likelihoods, rollout/hierarchy, VICReg/SIGReg,
                         displacement and counterfactual outcome objectives
  training/              trainer, optim/schedules, loggers
  probing/               feature extraction, linear probes, probe task registries
  planning/              closed-loop latent planners + symbolic diagnostics
scripts/                 train.py / probe.py / plan.py (all hydra)
tests/                   fast CPU tests for data, models, planning
```

Open-closed: new losses/models/datasets are new modules + config entries;
`CompositeObjective`, hydra `_target_`s and the probe/planner registries mean
nothing existing is modified.

## Usage

```bash
# discourse track (Hydra experiment configs)
.venv/bin/python scripts/train.py +experiment=disc_base
.venv/bin/python scripts/probe.py ckpt=runs/disc_base/best.pt
.venv2/bin/python scripts/plan.py ckpt=runs/disc_base/best.pt slack=0 lookahead=1

# diagnostic only: depth > 1 uses the reference graph to enumerate future
# feasible actions and is labeled as an oracle-action diagnostic
.venv2/bin/python scripts/plan.py ckpt=runs/disc_base/best.pt slack=0 \
  lookahead=2 allow_oracle_future_actions=true

# edit track
.venv/bin/python scripts/train.py +experiment=edit_base
.venv/bin/python scripts/probe.py ckpt=runs/edit_base/best.pt
.venv2/bin/python scripts/plan.py ckpt=runs/edit_base/best.pt

# probabilistic sentence stream (no intent/action interface)
.venv/bin/python scripts/train.py +experiment=sentence_vjepa_stylized
.venv/bin/python scripts/audit_variational_counterfactual.py \
  --ckpt runs/sentence_vjepa_stylized/best.pt --prior-samples 64

# one cell of the faithful observed-action LDAD stability factorial
bash scripts/run_observed_ldad_cell.sh ema vic on cuda:0

# probabilistic-state counterpart with an observed intent action
bash scripts/run_observed_vjepa_ldad_cell.sh ema vic on cuda:0

# information-matched LMs: rank intent phrases, then observe outcomes
.venv/bin/python scripts/train_lm.py +experiment=lm_intent
.venv/bin/python scripts/train_sentlm.py +experiment=sentlm_latent_intent

# generated decision tables for the active controlled studies
.venv/bin/python scripts/recipe_report.py --glob 'disc_latent_goal_*' \
  --reference disc_latent_goal_h2_r1 --out runs/round2_screen.md
.venv/bin/python scripts/selector_decision.py
.venv/bin/python scripts/buildup_decision.py
.venv/bin/python scripts/report_matched_baselines.py
.venv/bin/python scripts/report_variational_architecture.py
.venv/bin/python scripts/report_variational_transfer.py
.venv/bin/python scripts/report_action_free_transfer.py
.venv/bin/python scripts/report_variational_rollout.py
.venv/bin/python scripts/report_selector_screen.py
.venv/bin/python scripts/report_preference_sweep.py
.venv/bin/python scripts/report_action_grounding.py

# closed-loop selector margins and failure strata
.venv/bin/python scripts/analyze_failures.py \
  ckpt=runs/disc_latent_goal_h2_r1/best.pt device=cuda:0
# the same command auto-selects the faithful planner for igsm_real checkpoints

# claim-driven NeurIPS-style presentation
(cd reports && pdflatex discourse_jepa_neurips.tex && \
  pdflatex discourse_jepa_neurips.tex)
```

## Headline diagnostics

- `probe_value_pred` / `probe_value_rollout`: linear decodability of the step's
  computed value from the *predicted* latent (the predictor never sees the
  outcome) — latent arithmetic, teacher-forced vs open-loop.
- `probe_op_delta`, `probe_necessary_delta`: action type and goal-relevance
  from `Δs` (Delta-JEPA transition geometry).
- Planning: success within the *optimal* step budget vs random-feasible and
  oracle policies. The main lookahead-1 protocol exhaustively ranks every
  currently feasible intent, and all consequence evaluation is latent.
  Lookahead greater than one is excluded from headline comparisons because
  proposing future feasible actions requires the reference dependency graph.
- LM controls are separated into information-matched intent policies, which
  rank the same feasible intent phrases as JEPA, and outcome-candidate
  diagnostics, which score rendered consequences and therefore receive
  privileged candidate information.
- Probes are run against a random-init encoder control (`probe.py
  random_control=true`).
