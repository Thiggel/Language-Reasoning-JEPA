# TextJEPA

> **Results of the first experimental cycle: see [RESULTS.md](RESULTS.md).**
> TL;DR: latent planning over reasoning steps reaches 84% success at strict
> optimal budget (random: 5.5%) once (a) Delta-JEPA displacement decoding,
> (b) frozen-anchor next-chunk-embedding targets, and (c) a value-shaped
> encoder are combined.

Joint-embedding predictive architectures for **language**: latent world models
over discourse, with actions, hierarchy, and latent-space planning — no text
reconstruction anywhere in the JEPA core.

Two tracks share one latent-dynamics core:

| track | world state | action | "generation" |
|---|---|---|---|
| **discourse** | compressed state of the reasoning so far | tiny code of an intent phrase ("derive X from A plus B") | choose next reasoning step by latent search |
| **edits** | latent slots over a draft text buffer | span edit intent (delete/insert/replace) | edit the buffer until perfect |

## Why this setup

JEPA needs a *world* whose consequences are non-trivial to predict. Here the
world is an iGSM-style synthetic reasoning environment (a DAG of named
quantities with mod-p arithmetic): action phrases state **intent only**, so the
predictor must compute the *consequence* (e.g. the modular-arithmetic result
stated by the next step) purely in latent space. Because the data is synthetic
we have exact ground truth for every probe (values, resolved sets, remaining
steps, defect counts) and objective planning success criteria.

The core follows three papers:

- **I-JEPA / V-JEPA**: EMA target encoders, latent prediction losses, no
  reconstruction (arXiv:2301.08243).
- **Delta-JEPA** (arXiv:2606.31232): decode the executed action from the
  latent displacement `Δs = s_{t+1} − s_t` (op class + EMA phrase embedding).
  Displacement-level supervision prevents adjacent-state collapse and makes
  transitions action-sensitive for planning.
- **HWM, Hierarchical Planning with Latent World Models** (arXiv:2604.03208):
  macro-action encoder (CLS transformer over K action codes, deliberately tiny
  latent), high-level predictor in the *same* latent space, teacher-forcing +
  open-loop rollout losses, MPC-style planning with value/goal energies.

Stabilization: VICReg variance/covariance on online states + EMA targets.
The value head (predicted remaining steps / defects) is trained on **detached**
states by default so the state geometry is shaped only by JEPA losses.

## Layout

```
configs/                 hydra config tree (data / model / objective / probe / plan)
src/textjepa/
  data/igsm/             problem DAGs, NL rendering, symbolic env, dataset
  data/edits/            corruption + repair trajectories, edit env, dataset
  models/                encoders, state models, action bottlenecks (opt. FSQ),
                         predictor, Delta-JEPA decoder, EMA, shared dynamics core
  objectives/            latent pred, rollout, hierarchy, VICReg, delta-action, value
  training/              trainer, optim/schedules, loggers
  probing/               feature extraction, linear probes, probe task registries
  planning/              latent MPC planners + symbolic baselines
scripts/                 train.py / probe.py / plan.py (all hydra)
tests/                   fast CPU tests for data, models, planning
```

Open-closed: new losses/models/datasets are new modules + config entries;
`CompositeObjective`, hydra `_target_`s and the probe/planner registries mean
nothing existing is modified.

## Usage

```bash
# discourse track
python scripts/train.py run_name=disc_base train.epochs=20
python scripts/probe.py ckpt=runs/disc_base/best.pt
python scripts/plan.py  ckpt=runs/disc_base/best.pt slack=0 lookahead=1

# edit track
python scripts/train.py run_name=edit_base data=igsm_edit model=edit train.batch_size=64
python scripts/probe.py ckpt=runs/edit_base/best.pt
python scripts/plan.py  ckpt=runs/edit_base/best.pt
```

## Headline diagnostics

- `probe_value_pred` / `probe_value_rollout`: linear decodability of the step's
  computed value from the *predicted* latent (the predictor never sees the
  outcome) — latent arithmetic, teacher-forced vs open-loop.
- `probe_op_delta`, `probe_necessary_delta`: action type and goal-relevance
  from `Δs` (Delta-JEPA transition geometry).
- Planning: success within the *optimal* step budget vs random-feasible and
  oracle policies; the planner sees only the action interface, all
  consequence evaluation is latent.
- Probes are run against a random-init encoder control (`probe.py
  random_control=true`).
