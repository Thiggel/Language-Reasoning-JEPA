# Implementation audit

Four adversarial passes were completed before the corrected frozen diagnostic.
Each confirmed defect received a regression that failed before the repair.

## Pass 1: Stage 1 and data

- Fixed the objective from `NTP + λ_pred(cos + λ_scale scale)` to the specified
  `NTP + λ_pred cos + λ_scale scale`.
- Reinitialize the validation sampler on every evaluation so checkpoint
  comparisons use identical examples.
- Added packed position IDs that isolate documents in attention, while retaining
  a separate transition mask. WikiText paragraphs are grouped into articles to
  avoid destroying ordinary discourse context.
- Retained FP32 trainable predictor and LoRA weights under FP16/BF16 backbone
  execution.
- Made both no-action and action-only predictors capacity matched to full by
  retaining zero-valued missing-information channels.

## Pass 2: recurrence

- Stage 2 accepts either Stage 1 or the preceding Stage 2 checkpoint, making the
  `1 -> 4 -> 8 -> 16 -> 32` curriculum real rather than five restarts.
- Upper-stack steps accept per-example document-relative positions and an active
  key mask; cached states from preceding packed documents are excluded.
- Generated replay retains 64–128 jump states and supervises those visited
  states against an exact replay. The earlier code generated only `H+1` tokens
  and then restarted from an exact state.
- Added jump-only, periodic-refresh, exact speculative-verification, throughput,
  cache-growth, and perturbation-growth evaluation paths. Evaluation rollout
  runs under `no_grad`.

## Pass 3: goal geometry and planning

- Reasoning bundles now reject empty records, invalid terminals, width mismatch,
  non-boolean masks/labels, duplicate IDs, and non-finite tensors.
- Prompt states use offsets from the joint prompt-plus-trajectory tokenization,
  preventing BPE-boundary leakage.
- Oracle evaluation implements another-correct-same-problem,
  same-answer-different-problem, random-terminal, and relative-position controls.
- Batched beam search retains beams per root instead of applying one global
  top-k. Jump planning samples actions and advances leaves on the recurrent path,
  then commits one chunk and replans.
- Euclidean and directional quasimetric goal models are executable. Reports
  compare distance, direct value, and hybrid scores and include a distance
  Bellman residual over the actual candidate support.
- Reward construction selects one fixed global shaping coefficient. The previous
  per-trajectory future-dependent rescaling did not represent a fixed potential.

## Pass 4: execution

- Direct Grünau launch records `gruenau-gpus` and refuses a selected device unless
  both memory and utilization pass admission immediately before launch.
- Direct and Slurm jobs create terminal state, exit, provenance, environment, and
  fallback summary artifacts on failure. Slurm termination is trapped and a
  missing command or invalid snapshot is rejected before execution.

The focused suite contains 34 predictive-state and operational regressions.
The final audited snapshot passed all 771 repository tests (215 pre-existing
warnings). Repository-wide verification is required again after any change to
these interfaces.
