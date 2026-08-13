# Why nested-MPC planning returns zero: the worker's context, not its capability

_2026-08-13. Track: token_igsm. All planning numbers below are oracle-terminal,
i.e. **candidate-privileged diagnostics**, not deployable accuracy._

## Summary

The full-episode nested-MPC pilot (Alex job 3977693) returned 0/8 for all five
worker searches. That number turned out to be uninformative, and chasing it
further would have been wasted GPU. Three findings, in order of how much they
constrain the next decision:

1. **The pilot's accuracy metric was measuring termination, not reasoning.**
   No episode ever emitted a `\boxed{}` answer, so accuracy was 0 by
   construction.
2. **Four trained variants fail identically** at ~5% worker executability,
   invariant to metric, regularizer and learning rate — so nothing upstream of
   the worker is the binding constraint.
3. **A new 2x2 localizes the failure to context format.** Proposal validity
   collapses 0.727 -> 0.000 when only the prefix changes from canonical to
   self-generated. Instructing the model to use the right format recovers 0.039
   of that. Format is inherited from context, not from instructions.

The charter's second falsifiable decision — "whether a token worker can
realize a true future sentence waypoint" — is answered **no for a frozen
zero-shot Qwen**, and the reason is now a mechanism rather than a number.

## 1. The pilot metric was unmeasurable

Alex job 3977693 (`2026-08-10-faithful-nested-mpc-search-v5`) FAILED after
2h43m. All five worker cells wrote results; the job then died in the geometry
export on `AttributeError: 'SquaredEuclideanMetric' object has no attribute
'whiten'` (fixed, see §5).

| worker | accuracy | gen-failure | mean model cost error | wall/episode |
| --- | --- | --- | --- | --- |
| one_shot | 0/8 | 0.25 | 10.6 | 32 s |
| autoregressive_cem | 0/8 | 0.125 | 19.2 | 40 s |
| markov_cem | 0/8 | 0.0 | 23.3 | 27 s |
| factorized_cem | 0/8 | 0.0 | 39.7 | 24 s |
| beam | 0/8 | 0.0 | 50.7 | 1012 s |

Accuracy is `final_answer_matches`, which requires a `\boxed{N}` in the
generated text. **0 of 40 episodes emitted one.** Every trajectory was cut off
mid-preamble by the 8-step budget, where a "step" is one newline-delimited
segment and Qwen's zero-shot markdown spends its lines on headers and bullets.
Total generated text per episode was 62–210 tokens.

The 2026-08-05 gate evals confirm this is a termination metric: at
`max_reasoning_steps=16`, exactly **1/32 episodes reached a `\boxed{}`, and
that one episode is the one scored correct** (oracle-k1 and oracle-k4 both
1/32, mean generated steps 15.94 of 16).

Secondary observation, worth keeping as a figure: model cost error (exact
re-encoded cost minus predicted cost of the selected step) grows monotonically
with search strength, and text quality degrades with it — `one_shot` produces
fluent LaTeX reasoning, `beam` produces `"To solve这个问题…x stands after x"`.
With `worker_objective=jepa` there is no likelihood term at all, so this is a
clean demonstration of a planner exploiting a learned world model off-support.

## 2. Four variants fail identically — nothing upstream is the constraint

Every training run in this track ended in `validity_gate_stop`; no checkpoint
has ever passed. The four value-continuation variants
(`2026-08-05-nested-grounded-value-continuations-alex-s1-recovery-v6`):

| variant | worker_executability | oracle_success_gain | optimizer_curse_regret |
| --- | --- | --- | --- |
| euclidean-sigreg | 0.0498 | 0.0 | 24.66 |
| euclidean-vicreg | 0.0601 | 0.0 | 10.00 |
| euclidean-vicreg-lr1e4 | 0.0537 | 0.0 | 9.54 |
| mahalanobis-vicreg | 0.0469 | 0.0 | 2.17 |

Executability (fraction of macro-action rollouts that are symbolically
verified) sits in a tight band around 5% while curse regret swings by a factor
of 11. Gate threshold is 0.10. `oracle_success_gain = 0` in all four: raising
manager lookahead K1 from 1 to 4 buys nothing.

Meanwhile the **sentence level passes its gates** on the same checkpoints:
`symbolic_state_purity 1.0`, `sentence_effective_rank 23.6`,
`exact_symbolic_waypoint_success 1.0` (predicted 0.859),
`heldout_sentence_dynamics_gain 0.967` vs identity, commutation error 0.104 vs
baseline 0.176 (41% gain), `worker_success 0.891` in the commutation gate.

Reading: the manager knows where to go and the sentence world model is sound.
The worker cannot write text that gets there.

## 3. The 2x2: format is inherited from context, not instructions

Round `2026-08-12-worker-context-coverage-v1`, run on Grünau (gruenau7, A6000),
128 roots, population 32, K0 64, seed 0. Frozen-reference proposal coverage
only: **no JEPA checkpoint is loaded and no planner runs**, per the contract's
labelled-ablation clause for the Frozen reference section.

- **prefix** = the context the worker proposes from. `canonical` = the
  teacher-forced ground-truth solution prefix; `self_generated` = the model's
  own free-run text, which is the regime the worker is actually in during MPC.
- **prompt** = `pinned` is the contract's frozen-reference system prompt;
  `canonical` is a new one demanding the iGSM step form (added as
  `CANONICAL_SYSTEM_PROMPT` / `canonical_prompt_token_ids`).

| prefix | prompt | parse | greedy | oracle@8 | oracle@32 |
| --- | --- | --- | --- | --- | --- |
| canonical | pinned | 0.417 | 0.422 | 0.641 | **0.727** |
| canonical | canonical | 0.514 | 0.266 | 0.570 | **0.742** |
| self_generated | pinned | 0.000 | 0.000 | 0.000 | **0.000** |
| self_generated | canonical | 0.104 | 0.008 | 0.023 | **0.039** |

`unreachable_rate = 0.000` in every cell: the model always produced enough
steps to reach the requested depth. It is not stopping early.

**The decisive comparison is the `oracle@32` column.** Changing only the prefix
collapses the worker from 0.727 to 0.000. Changing only the prompt, at a
self-generated prefix, recovers 0.039 — about 5% of what was lost. Context
beats instruction by roughly an order of magnitude.

The `self_generated x pinned` zero is categorical, not merely small: **0 of
128 x 32 = 4096 continuations** parsed as a canonical operation. Direct
inspection confirms the mechanism rather than a bug:

```
reference step:   'so the number of purple hats is 18 minus 6 = 12 .'
free-run step 0:  'Here is the step-by-step logical deduction to solve the problem:'
                  '**1. Identify the given values:**'
free-run step 1:  '*   Number of round knots ($R$) = 18'
continuations:    '*   Number of hard nails ($H$) = 6'   (x4, none parse)
```

This is **not** an error-recovery failure. `round knots = 18` is a correct
given; the content is right and the notation is wrong. The model's first
free-run line is a markdown header, and from there its own context keeps it in
markdown indefinitely. It never enters the required track at all.

The notation matters twice: `parse_rendered_operation` only accepts the word
forms `plus|minus|times`, so `($R$) = 18` can never be scored; and more
fundamentally the JEPA states were built by encoding canonical sentences, so
off-format text lands the system in latent states the predictor never saw.

## 4. Caveats

- `greedy` / `oracle@1` is the validity of **sample 0 at temperature 0.8**, not
  argmax decoding — a single Bernoulli draw per root, SE ~0.044 over 128 roots.
  Treat `oracle@32` as the robust column.
- The control reproduces the historical gate's `oracle@32` **exactly** (0.727 vs
  0.7265625, the same 93/128 roots) but its `greedy` differs (0.422 vs 0.508,
  ~2 SE). Likely cause: Grünau runs torch 2.5.1 on an A6000, Alex ran torch
  2.9.0; sampling RNG is not bit-portable. All four cells ran in the same
  environment, so the within-2x2 comparisons are internally consistent.
  A cross-cluster replication was attempted twice on Alex (jobs 3998023,
  3999448) and failed at the node level in <10 s with no output; not pursued.
  Open alternative: rerun the control at 2–3 seeds on Grünau.
- "The content is right at self-generated prefixes" is an **inference** from the
  inspection above plus the 0.73 at canonical prefixes, not a measurement:
  nothing parses there, so content cannot be scored. Constrained decoding is
  the experiment that tests it.
- The canonical prompt trades form for content: at a matched canonical prefix
  it raises parse 0.417 -> 0.514 but halves single-sample validity
  0.422 -> 0.266, and barely moves `oracle@32`. It is not a free win.
- `mean_worker_exact_gap` is 0.0 by construction for the search workers (they
  return a single winner), so selection regret is only informative for
  `one_shot`.

## 5. Code changes made in this round

- `SquaredEuclideanMetric.whiten` added with a non-persistent `mean` buffer, so
  the geometry export works under either metric and existing checkpoints still
  load with `strict=True`. Regression test added.
- `run_hierarchical_mpc_search_matrix.py` had `--manager-grounding none`
  **hard-coded**, which is why the contract violation ("grounded costs, not
  merely model-predicted costs, select subsequent CEM elites") was invisible in
  configs. Now a flag defaulting to `shared_bank`.
- `mean_optimizer_curse_regret` reported `0.0` when nothing was grounded (NaN
  count 0 divided by `max(1, ...)`), reading as "no curse" instead of "not
  measured". Now emits `null` plus `optimizer_curse_regret_samples`.
- Added `final_answer_rate`, `step_parse_rate`, `step_validity_rate` to the MPC
  evaluator so accuracy is no longer a pure termination metric.
- `evaluate_hierarchical_proposal_coverage.py` gained `--prefix-source` and
  `--prompt-style`; defaults reproduce the previous gate path byte-identically.
- Suite: 845 passed, 0 failed.

## 6. Where this leaves the track, and what to run next

Ruled out as the binding constraint: the JEPA representation (gates pass), the
metric/regularizer/learning rate (four variants identical), planner compute
(K1=4 buys nothing over K1=1; beam spends 269k transition evaluations for the
worst text), and prompting (this 2x2).

Not yet ruled out, in priority order:

1. **Prefix priming** (~30 min, frozen LM untouched, charter intact). Plant one
   canonical line at the head of the assistant turn instead of instructing in
   the system prompt. Format follows context, and a canonical context yields
   0.42–0.51 parse; if the planner's accepted steps stay canonical the
   trajectory is self-reinforcing. Design note: the seed line must be labelled —
   teacher-forcing it makes the run privileged; templating it from the problem
   text (which contains the variable names) is the deployable version.
2. **Constrained decoding** onto the canonical grammar. Format becomes 100% by
   construction and validity then depends purely on content. This is the
   decisive test of the §4 inference: parse -> 1.0 with validity climbing toward
   0.73 means the whole 5% wall was a formatting artifact; parse -> 1.0 with
   validity near 0 means the content genuinely is not there off the
   teacher-forced path.
3. **Fine-tuning the generator** would likely work but changes the charter's
   question, since "derived from a frozen reasoning LM" is constitutive of it.
   Hold in reserve until 1 and 2 are decided.

Pair 1 with a parse-gate on worker candidates — the "supported proposals" idea
already in `FULL_HIERARCHICAL_LANGUAGE_EXPERIMENT.md` — so the JEPA cost cannot
select an off-manifold step and derail the episode.

## Artifacts

- 2x2: `runs/autonomy/token_igsm/2026-08-12-worker-context-coverage-v1/gruenau-id_test-2x2/`
  (`cells/*.json`, `summary.json`, `job.sh`)
- Pilot: Alex `runs/autonomy/token_igsm/2026-08-10-faithful-nested-mpc-search-v5/`
- Gates: Alex `runs/autonomy/token_igsm/2026-08-05-nested-grounded-value-continuations-alex-s1-recovery-v6/*/gates/`
- Snapshot `d9a4bfbd901a55dfc60a35609d28cbd3a6604f7f` on Alex (base commit
  `43b6cbe` + `SNAPSHOT_UNCOMMITTED.patch`).
