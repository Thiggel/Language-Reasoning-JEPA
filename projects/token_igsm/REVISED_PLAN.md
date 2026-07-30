# Hierarchical predictive-state planning for language

This is the active specification for the former discourse/token-iGSM line as
of 2026-07-30. Historical experiments remain evidence about earlier
architectures; they are not evidence for this revised architecture.

The later [normative implementation contract](NORMATIVE_CONTRACT.md)
supersedes any ambiguous or conflicting passage here, especially the strict
nested encoder structure, notation, boundary indexing, task embedding,
pre-action context, oracle-goal naming, and frozen-LM constants.

## Core hypothesis

A frozen causal reasoning LM supplies wide hidden states, executable token
proposals, counterfactual branches, and exact branch re-encodings. A nested
tower learns increasingly coarse controlled predictive spaces:

```
task completion -> sentence waypoint -> token realization

x_<n -> E_LM -> h_n -> E0 -> z_n^0 -> E0_to_1 -> z_j^1
```

`E0_to_1` is the renamed coarsening map formerly called `C0`. There is no
parallel direct sentence projector in the core model.

The first question is deliberately narrower: can an action-conditioned
predictive representation support oracle-goal planning?

## Notation and information boundary

- `K0`: token actions imagined by the worker.
- `K1`: sentence actions imagined by the manager.
- `KV`: maximum offline continuation depth for value targets.
- `n_exec`: actual tokens or steps executed before exact re-encoding and
  replanning.

There is no remaining-budget input. The eventual value is
`V_eta(xi1, q)`. It receives the sentence predictor's contextual state and a
task embedding, never an answer, terminal embedding, or search-depth input.

Symbolic iGSM state is evaluation-only. Oracle terminal states and true future
waypoints must be labelled oracle/candidate-privileged in every artifact.
Cross-project information is disabled.

## Stage gates

| Stage | Added mechanism | Gate to continue |
| --- | --- | --- |
| 0 | Frozen-LM proposal coverage | Useful branches have measurable oracle@N coverage |
| 1 | Token encoder `E0`, EMA target, causal predictor `P0` | Token prediction and causal-state gates pass |
| 2 | Flat token-space oracle endpoint | Exact/model separation succeeds for `k={4,8,16}` |
| 3 | Nested `E0_to_1`, text-only `A1`, and sentence predictor `P1` | Sentence endpoint diagnostics pass |
| 4 | Oracle next-sentence waypoint worker | Exact symbolic waypoint success improves with `K0` |
| 5 | Optional dynamic commutation | Requested and achieved coarse outcomes agree |
| 6 | Gaussian posterior/prior over 32-dimensional innovations | Prior-coordinate samples are supported and worker-executable |
| 7 | Oracle-terminal high-level planning | Exact and learned success improve with compute-matched `K1` |
| 8 | Unbudgeted value ranking distillation | Low top-one regret without terminal input |
| 9 | Full hierarchical MPC and grounded replay | ID/OOD gains survive exact execution and optimizer-curse tests |

Value, latent macro-actions, VQ, temporal straightening, logit reconstruction,
and backbone fine-tuning are absent from the initial experiment. Dynamic
commutation is disabled until each level independently predicts its actual
encoded endpoint.

## Initial implementation

- `E0`: two-layer RMSNorm/SiLU token encoder, dimension 256.
- `P0`: residual causal token predictor, width 512, four blocks, eight heads,
  bounded context 64 with random truncation and cache dropout.
- `E0_to_1`: two-layer nested sentence encoder, dimension 128.
- `P1`: residual causal sentence predictor, width 512, three blocks, eight heads,
  bounded context 32.
- `A1`: text-only observed-span encoder, action dimension 32.
- EMA target copies of `E0` and `E0_to_1`, with stop-gradient targets.
- `Pi1`: later causal prior over high-level macro-actions.
- `V`: later unbudgeted top-level cost-to-completion.
- VICReg squared variance floor and covariance decorrelation at each active
  level.
- EMA covariance, isotropic shrinkage, and Mahalanobis endpoint discrepancy.
- Rectangular root-by-candidate branch batches and append-only planner replay.

The code consumes offline exact hidden states rather than owning the frozen
0.8B pilot LM (or a later gated 7B LM). This preserves the required separation between expensive generation
or exact reanalysis and ordinary gradient-bearing learner batches.

The first Qwen3.5-0.8B pilot intentionally uses non-thinking chat mode because
the verified iGSM reasoning trace is appended externally. It uses
`top_p=.95`, `top_k=0`, no presence penalty, and records the optimized/fallback
kernel backend. Thinking mode is not silently mixed into the same proposal
coverage curves.

Implementation:

- `src/textjepa/models/hierarchical_language_jepa.py`
- `src/textjepa/objectives/hierarchical_language.py`
- `src/textjepa/planning/hierarchical_language.py`
- `src/textjepa/data/hierarchical_language.py`
- `src/textjepa/training/hierarchical_language.py`
- `src/textjepa/analysis/hierarchical_language.py`
- `scripts/train_hierarchical_language_jepa.py`
- `configs/experiment/hierarchical_language_oracle_token.yaml`
- `configs/experiment/hierarchical_language_oracle_waypoint.yaml`
- `configs/experiment/hierarchical_language_full.yaml`

## Data and evaluation

iGSM is primary. Realized generator statistics determine exact length
buckets; the intended split axes are ID, near length-OOD, far length-OOD,
structural OOD, and paraphrase OOD. GSM8K is evaluated first as transfer and
then with official train/test adaptation. Verified OpenThoughts math is gated
until the two decisive oracle experiments succeed.

Every major result varies `K0`, `K1` where active, population, iterations,
explicit horizon, and execution interval. Candidate-level records retain
predicted, exact, achieved, symbolic, prior, search-iteration, FLOP, latency,
and provenance fields. Compute-matched comparisons and optimizer-curse curves
are mandatory.

All additions also report fractional estimated FLOPs and wall time by
component: observed frozen-LM encoding, counterfactual generation, exact
re-encoding, dense JEPA training, counterfactual replay, value distillation,
macro prior evaluation, transition rollout, objective evaluation, and exact
grounding.

## Legacy boundary

Earlier records are immutable legacy:

- [`research/hard_text/`](../../research/hard_text/README.md)
- [`research/hard_text/logs/`](../../research/hard_text/logs/README.md)
- historical `hard_hier_*`, `text_hier_*`, and `deltajepa_text_*` run
  families
- existing figures, checkpoints, controller rounds, reports, and wave notes

They remain at their original paths so log-derived insights and provenance
continue to resolve. New artifacts must not be written into those historical
run families.
