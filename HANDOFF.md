# HANDOFF — complete state of the TextJEPA project

_Last updated: 2026-07-13 (all numbers verified against `runs/` artifacts)._
This document is the single entry point for continuing this project. Read it
together with `RESULTS.md` (all findings, numbered) and the two beamer decks
in `reports/` (every experiment explained with claim-driven tables/figures;
`reports/discourse_jepa.pdf` slide 4 is the glossary for all terminology).

---

## 1. What this project is

Two JEPA-pure (no reconstruction) latent world models over language:

- **Discourse track** (`disc_*` runs): a reasoning trace is a trajectory —
  state `s_t` = compressed discourse state, action = *intent phrase* (never
  contains the outcome), predictor `F(s,a)` must compute consequences in
  latent space. Generation = closed-loop MPC over the environment's
  discrete feasible-action interface, scored by a learned energy.
- **Edit track** (`edit_*` runs): state = text buffer (draft solution),
  actions = delete/insert/replace edits, "edit until perfect".

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
# planning variants
.venv2/bin/python scripts/plan.py ckpt=runs/X/best.pt slack=0 lookahead=2 \
    [energy=oracle_goal] [hierarchy=true] [+simulator=symbolic]
# counterfactual audit (matching / Kendall-tau / RSA vs symbolic truth)
.venv/bin/python scripts/audit_counterfactual.py ckpt=runs/X/best.pt n_episodes=100
# LM baselines: scripts/train_lm.py + scripts/plan_lm.py (parity selection)
# sentence-LM baselines: scripts/train_sentlm.py + scripts/plan_sentlm.py
# variational planner: scripts/plan_var.py (samples codes from the prior)
# faithful fidelity gate: .venv/bin/python scripts/validate_faithful.py 60
# figures + report: .venv2/bin/python scripts/make_report_figs.py; scripts/report.py
# decks: cd reports && pdflatex discourse_jepa.tex (twice)
# tests: .venv/bin/python -m pytest tests/  (17 tests, all green)
```

Known hydra pitfalls: keys that already exist in a config must be set
WITHOUT `+` prefix; experiment files with a `defaults:` list need it as the
first block after the `# @package _global_` header.

## 4. Where the science stands (headline numbers)

Stylized easy domain, success @ strict optimal budget / @ slack-2
(3-seed mean±std where given; oracle = 1.0, random = 0.055/0.405):

| recipe | @opt | @slack2 |
|---|---|---|
| token LM (matched params, best lr) | 0.670 | 0.910 |
| **core5 (THE sheared recipe)** | **0.880±0.004** | **0.975±0.004** |
| + cost ranking, look-2 (`mdr_cal`) | 0.945 | 1.00 |
| annotation-free tier (pre-GAR best) | 0.30–0.36 | 0.72–0.78 |

**core5** = latent prediction + counterfactual ranking + VICReg + frozen
anchor + LDAD; value head trained by ranking margins (no scalar labels).
Found by two-directional shearing (see deck Result 16). Key facts a new
contributor must know:
- **Ranking is the single load-bearing energy ingredient** (0.64→0.91) —
  but as currently implemented it uses SYMBOLIC counterfactual outcome
  labels. Replacing this is the active work item (see §6).
- Depth: uncalibrated ranking collapses with lookahead (esp. hard domain:
  0.22→0.01); `cost_ranking` fixes it but only on fuller recipes.
- Stability: stopgrad+VICReg suffice; EMA optional (+0.03); LDAD alone
  does NOT prevent collapse (Delta-JEPA stability claim refuted here).
- Hierarchy loss: removable; F_hi useful only at plan time for
  uncalibrated models.
- Full audit trail: `runs/*/counterfactual_audit.json` (F(s,a) matching
  1.00 vs 0.30 shuffled control on trained models).

Hard stylized: `hard_rank` 0.19±0.02 @look1; `hard_rank_cal` 0.255 @look4
(= hard best). LM: 0.01. Faithful iGSM: random 0.20; combo 0.245
(energy-limited, probes healthy); `real_rank` **0.410/0.735**;
`real_rank_big` (35M) 0.460/0.765; official-hard `real_rank_hard`
0.385/0.650. Edit track: `edit_attn` (attention predictor over buffer
sentences) = record 0.510/0.550; audit matching capped ~0.50 by the slot
ENCODER (both audit sides use it) — per-sentence encoding is the known fix.
Variational unobserved actions: v2 (`disc_var_fb`, free-bits) 0.195/0.680.
LM fairness: DPO-style ranked LM does NOT close the gap (0.565–0.655 vs
plain 0.670 vs JEPA 0.91) — the advantage is architectural.

## 5. IN FLIGHT right now (check these first)

| run | where | what it answers |
|---|---|---|
| `disc_georank` | gruenau12 gpu0 | **GAR**: annotation-free ranking (see §6) with label-free hinge + distilled V |
| `disc_georank_pure` | gruenau12 gpu1 | GAR alone (V from geometric margins only) |
| `real_rank_k1` | gruenau11 gpu0 | faithful counterfactual-density sweep (K=2 was 0.410) |
| `real_rank_k4` | gruenau11 gpu3 | ditto, K=4 (stylized saturated at K=4) |

Each chain = train → `eval_run.sh` → (georank runs) audit; completion
marker `DONE_<name>` appended to `runs_loop.log`. ~1–2h from the timestamp
above. No other jobs are running anywhere.

## 6. THE ACTIVE RESEARCH QUESTION (read this before doing anything)

The user's directive: **eliminate symbolic supervision from the recipe**,
because any symbolic ingredient (a) weakens the claim that JEPA itself
solved the task and (b) blocks transfer to real language. Supervision
taxonomy now used everywhere:
(a) trace-only (text + geometry — transfers to real language),
(b) environment interaction with GEOMETRIC labels (transfers if you can
    sample counterfactual continuations),
(c) symbolic annotations (counts/flags — do NOT transfer).

Just implemented: **GAR (Geometric-Advantage Ranking)**,
`GeoAdvantageRank` in `src/textjepa/objectives/ranking.py` + `_geo_rank`
in `models/discourse_jepa.py` + `geo_rank_k` in the dataset: at ONE anchor
step per trace, the env executes K alternative actions (returning outcome
TEXT only); the EMA teacher encodes those true next states; their LN-L1
distance to the EMA terminal goal orders `V(F(s,a_i))` via a margin loss
(label_gap 0.02 filters geometric ties). Tier (b): no annotations.

**Decision tree when the in-flight runs land:**
- `disc_georank*` ≈ symbolic ranking (≥0.8): headline result. Then (1)
  3 seeds; (2) port GAR to faithful (`FaithfulDataset` needs geo_rank_k —
  mirror the igsm implementation; the official env renders alt sentences
  via `FaithfulEnv.clone().step()`); (3) multi-step GAR (k-step env
  rollouts of alternatives — user explicitly wants this); (4) reframe the
  deck's supervision ladder around tier (b).
- GAR lands mid (0.5–0.7): diagnose via the audit (is the GEOMETRIC label
  order itself wrong? correlate ga_label ordering with symbolic outcomes
  offline); likely fixes: shape the metric first (the hinge/straightening
  runs), larger K, label_gap sweep.
- GAR fails (<0.4): the geometry isn't a good enough teacher yet →
  strengthen label-free shaping first (see straightening results, deck
  Results 7/8), or self-ranking against imagined alternatives.

## 7. Next planned items (agreed with the user, in order)

1. **GAR verdict + follow-ups** (§6 decision tree).
2. **Uniform-stream variational v3** (design agreed, not yet built): treat
   intent sentences as ordinary observations in one stream; u_t = latent
   RESIDUAL per transition (posterior q(u|s_next) target-only, N(0,I)
   prior, free-bits ≥2 nats — v1 collapsed without them); F stays
   deterministic (regular JEPA loss; learned-σ NLL only as an ablation
   arm, per VJEPA arXiv:2601.14354 — see the VJEPA discussion in the
   conversation/memory). Planning WITHOUT any decode head: for each
   feasible intent i (interface text, outcome-free), encode
   s_i = state(history+intent_i), score E_{u~prior}[V(F(s_i,u))], pick
   argmin — discrete choice, no CEM, no off-manifold risk. Pure-discovery
   variant (no intents at plan time): compare decode-head vs
   nearest-imagined-state matching (user wants BOTH tried).
   Evidence suite: frozen-u probes (op/var classification, intent-anchor
   retrieval, taxonomy clustering) — "does u encode the intent".
   Existing machinery to reuse: `VariationalAction` (models/action.py),
   `ActionKL` (free_nats param), `act_decode`, `scripts/plan_var.py`.
3. **Per-sentence edit encoding (both sides)** — lifts the edit audit cap
   (encoder-side, ~0.50); `AttnEditPredictor` exists, the encoder is next.
4. Paper assembly: decks are the source of truth; the framing decision
   (annotation-free headline vs symbolic headline with the ladder) awaits
   the GAR verdict.

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
- The planner's `EpisodeResult`/plan JSONs: keys are
  `latent_planner[_oracle_goal]`, `lm_policy`, `sentlm_*`, `var_planner`.
  `scripts/report.py` aggregates everything into `runs/report.md`.
- Old checkpoints load with `strict=False` (new heads init randomly and a
  note is printed); config keys added later are read with `.get(...,
  default)` in `build_dataset`/`load_run` for backward compat.
- Deck compiles need two pdflatex passes; figures regenerate from run
  artifacts automatically (`make_report_figs.py` skips missing files).

## 9. Documentation map

- `RESULTS.md` — findings 1–21 + probe section + tables.
- `reports/discourse_jepa.pdf` (37+ slides) — full method + results;
  glossary slide 4; shear log Result 16; scale-up preview near the end.
- `reports/edit_jepa.pdf` (9 slides) — edit track.
- `runs/report.md` — auto-generated master table (energy × slack × look).
- `tests/` — 17 tests; run before committing model/objective changes.
