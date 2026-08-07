# Normative hierarchical-language implementation contract

This document supersedes ambiguous indexing, stage-ordering, goal, task-input,
encoder-structure, notation, and checkpoint passages in the earlier revised
plan.

## Canonical nested representation and notation

The operational hierarchy is strictly nested:

```text
x_<n --E_LM--> h_n --E0--> z_n^0 --E0_to_1--> z_j^1
```

`E_LM` is the frozen language model, `E0` is the token planning encoder, and
`E0_to_1` is the sentence planning encoder. Sentence states are instantiated
at boundaries:

```text
z_j^1 = E0_to_1(E0(H(B_j)))
```

There is no authoritative direct `h -> z1` encoder and no separate `C0`.
`C0` was the old name for `E0_to_1`. The first implementation does not include
the optional direct teacher ablation.

The authoritative sentence target is unambiguously the fully nested EMA path:

```text
bar_z_j^1 = bar_E0_to_1(bar_E0(H(B_j)))
```

It is never `bar_E0_to_1(E0(h))`, and no direct `h -> z1` target exists in the
core experiment.

`P0` and `P1` denote the token and sentence transition predictors. `A1`
encodes only the complete sentence token span, not the current state or LM
hidden sequence. `Pi1` is the causal macro-action prior and `V` is the
task-conditioned top-level cost.

## Frozen reference

- Model: `Qwen/Qwen3.5-0.8B`
- Revision: `2fc06364715b967f1860aea9cf38778875588b17`
- Transformers: `5.13.0`
- BOS: disabled
- Padding: `<|endoftext|>` / `248044`
- Primary EOS: `<|im_end|>` / `248046`
- Generation stop IDs: `248046`, `248044`
- System prompt: `Please reason step by step, and put your final answer within \boxed{}.`
- Thinking mode: explicitly disabled for the first externally rendered iGSM
  pilot (`enable_thinking=false`); thinking-mode results are a separately
  labelled proposal-coverage ablation.
- Step delimiter: newline
- Counterfactual step: 4–64 tokens

This 0.8B post-trained Qwen is the first mechanism/debugging backbone. Scaling
to the earlier 7B math model is gated on the flat-token and sentence-worker
validity checks. The collector is
[`scripts/collect_hierarchical_language_features.py`](../../scripts/collect_hierarchical_language_features.py).
It uses the pinned chat template, saves exact final-layer hidden states, and
can collect the specified eight-way next-step mixture.

## Prefix-length indexing

`H(n)` is the representation after exactly `n` tokens and is stored at hidden
tensor position `n - 1`. For prompt length `P` and solution end `E`:

```python
current = z[:, P - 1:E - 1]
actions = input_ids[:, P:E]
targets = target_z[:, P:E]
```

Global sentence boundaries satisfy:

```text
P = B0 < B1 < ... < BJ = E
```

The action span is `input_ids[Bj:Bj+1]`; the state is gathered at
`hidden[Bj - 1]`. Concatenating every action slice must reproduce
`input_ids[P:E]`. EOS and padding never become JEPA actions.

The required feature fields are `input_ids`, `attention_mask`, `prompt_len`,
`solution_end`, `step_boundaries`, `reasoning_depth`,
`canonical_state_ids`, `problem_id`, `template_family`, and `graph_family`.
Boundary padding is right-contiguous `-1` with an explicit mask.

## Causal state/action timing

The predictor constructs pre-action context `c_j` from the current state and
only preceding actions. The current action is supplied afterward to predict
the successor:

```text
c_j = Context(z_0, u_0, ..., u_{j-1}, z_j)
z_hat_{j+1} = Transition(c_j, u_j)
```

The macro prior and value read `(z_j, c_j, e_q)`. They cannot see `u_j`, the
true successor, or future text. The training posterior alone may see the
completed span and true endpoint.

`e_q` is a trainable projection of `hidden[prompt_len - 1]`. It is computed
once per problem and is invariant to the solution suffix.

## Three distinct oracle experiments

1. **Flat token endpoint.** Direct token-space target
   `g^0_{n,k} = target_E0(H(n+k))`, initially `k=8`, then `{4,8,16}`.
   Neither `E0_to_1` nor `P1` participates.
2. **Next-sentence waypoint.** At a real boundary, score complete
   newline/EOS-terminated candidate spans after mapping predicted token
   endpoints through the trained nested encoder `E0_to_1`.
3. **Verified terminal sentence set.** Use complete verified solution states
   only for oracle high-level planning and offline value-teacher construction.
   The student and inference planner never receive this set.

## Hierarchical search contract

High-level CEM predicts up to `K1` sentence transitions before applying the
oracle terminal discrepancy or deployable value. Its primary supported form
optimizes standard-normal coordinates decoded by `Pi1`; ambient-action CEM,
an explicit noise trust region, and additive prior NLL are separate ablations.

The primary token objective is only:

```text
distance(E0_to_1(P0^K0(current_token_state, candidate_tokens)), waypoint)
```

Qwen supplies causally supported next-token proposals but its likelihood is
not part of this primary score. Likelihood-only and small-NLL-addition controls
are reported separately. Full-prefix beam search and elite-prefix
autoregressive CEM are the faithful workers. Sparse first-order Markov CEM is
a weaker dependency ablation. Independent-position categorical CEM is an
explicit negative control and cannot be described as faithful language CEM.

Open-loop hierarchy executes the waypoint sequence originally imagined by one
manager call. Closed-loop hierarchy executes only the first waypoint, exactly
re-encodes the achieved sentence state and causal histories, and replans from
that corrected state. Both are reported; closed-loop is the primary MPC mode.

Within either hierarchy policy, token MPC reports `n_exec` independently.
For positive `n_exec`, the worker executes at most that many tokens, exactly
re-encodes the partial prefix, and replans toward the same sentence waypoint.
Only a complete newline/EOS span is passed to `A1` and counted as an achieved
sentence transition. Complete-sentence execution (`n_exec=0`) is a control,
not a substitute for token-level receding MPC.

## Stage order

| Stage | Active addition |
| --- | --- |
| 0 | Pinned data/indexing validation |
| 1 | Token encoder `E0`, EMA target, and token predictor `P0` |
| 2 | Flat token-space oracle planning (evaluation only) |
| 3 | Nested sentence encoder `E0_to_1`, `A1`, and sentence predictor `P1`, with admitted token modules loaded and frozen |
| 4 | Oracle next-sentence worker (evaluation only) |
| 5 | Dynamic commutation ablation |
| 6 | Macro posterior/`Pi1` training |
| 7 | Oracle-terminal high-level planning (evaluation only) |
| 8 | Multi-depth oracle value teacher and listwise value distillation |
| 9 | No-terminal hierarchical inference (evaluation only) |
| 10 | Exact worker-achieved replay and closed-loop reanalysis |

State consistency is definitional because the sentence state is
`E0_to_1(z0)`. Dynamic commutation is not a prerequisite for the worker. A
worker checkpoint must contain the trained nested encoder and held-out
sentence endpoint measurements.

The nested sentence admission gate reports held-out sentence dynamics,
sentence effective rank, paraphrase-versus-symbolic purity, and exact worker
endpoint agreement. There is no static self-alignment gate.

The later commutation ablation compares:

```text
E0_to_1(P0^|y|(xi0, y)) ~= P1(xi1, A1(y))
```

## Counterfactual and replay policy

Every selected real sentence root receives:

- observed continuation: 1
- greedy continuation: 1
- `T=0.5, top-p=.95`: 2
- `T=0.8, top-p=.95`: 2
- `T=1.0, top-p=.95`: 1
- `T=1.2, top-p=.95`: 1

Sampling is nucleus-only: `top_k=0`. No presence or repetition penalty is
applied. Sampling uses an explicitly seeded device RNG; no unsupported
`generator` keyword is passed to Transformers. A per-row newline stopping
criterion stops generation once a complete delimiter is reached after the
four-token minimum.

The default collector uses a lossless shared-cache execution engine. It
prefills each real root once, forks the complete hybrid attention/recurrent
cache across the seven generated slots, and removes completed rows from the
active decoding batch. A newline before four tokens is retired immediately
because the stored candidate was already defined by trimming at that first
delimiter. The legacy `model.generate` engine remains available for matched
audits. Engine identity and backend availability are stored in provenance.
The configured generation batch size is a hard upper bound for both root
prefills and branch-decode forwards, including values smaller than seven.
Terminal tokens sampled from prefill or prior-step logits are recorded without
an unnecessary successor forward.

Newline-complete candidates train token and sentence dynamics. EOS candidates
are terminal sentence actions. Length-truncated candidates train token
dynamics only. Exact suffix hidden states, candidate source, temperature,
termination mode, and root state are stored.

Cached autoregressive hidden states are not accepted as exact JEPA targets.
Qwen's recurrent cached path can differ numerically from a complete
teacher-forced pass, so every observed and generated branch is independently
re-encoded with `use_cache=False`. Shared-cache generation is only the
proposal mechanism.

Feature and counterfactual artifacts bind the numerical dtype and dense
encoding batch size. Counterfactual artifacts additionally bind the seed,
generation batch size, re-encoding batch size, and collection engine; resume
rejects any mismatch. Generation FLOPs count the padded tensor positions
actually submitted to model forwards, not only attention-valid tokens.

Production-scale collection is a bounded streaming operation. The top-level
`.pt` artifact is a provenance manifest whose checksummed feature parts contain
at most 256 observed examples and whose counterfactual parts contain at most 16
root examples. Parts are atomically committed and may be resumed independently.
The learner randomizes part order and example order but holds only one part in
host memory at a time. A manifest fingerprint binds the ordered part hashes and
counts, so sharding changes storage and scheduling—not the admitted dataset.
Small tests may retain the backward-compatible monolithic artifact format.

Sparse recursive rollout uses `N=1` always, `N=2` with probability `.25`,
`N=4` with `.05`, and `N=8` with `.01`; horizons above two truncate BPTT
every two transitions while preserving the root predictor cache.

## Joint two-level training and collapse control

The token-only model remains an information-matched diagnostic and control; it
is not the initialization path for the primary two-level model. The primary
strict-nested model starts from scratch and jointly updates `E0`, `P0`,
`E0_to_1`, `A1`, and `P1`. Its targets remain the fully nested EMA path and are
stop-gradient. This joint objective is the intended source of both local token
control information and reasoning-step temporal abstraction.

Every optimizer step that updates an online encoder also applies anti-collapse
regularization to the states produced on that step. This includes dense traces
and counterfactual replay. Joint replay applies the regularizer to both token
states and sentence-boundary states; counterfactual sampling is not exempt from
the representation objective merely because it provides additional actions.

The clean collapse-repair reference uses coordinate-mean squared Euclidean
prediction, with VICReg weights scaled consistently with that reduction
(`prediction=25`, `variance=25`, `covariance=1`) at both levels. Normalized
Mahalanobis is a separate geometry ablation rather than the default training
loss. A SIGReg cell uses the official sliced Epps--Pulley statistic (17
integration points and 256 random projections) with a `0.95/0.05`
prediction/regularization mixture; it retains EMA and stop-gradient for matched
architecture and is therefore labelled a SIGReg regularizer ablation, not a
complete LeJEPA reproduction.

## iGSM and transfer splits

Depth is verified symbolic operation count excluding final-answer emission:

- train/ID/structural/paraphrase: 2–6
- near length-OOD: 7–9
- far length-OOD: 10–12

Reference counts are 200k train; 25k each ID validation, ID test, structural
OOD, and paraphrase OOD; and 15k each near/far length-OOD. Split assignment
and balanced manifest construction are implemented in
`textjepa.data.language_planning`. GSM8K and verified OpenThoughts traces use
the same prompt/feature representation, with natural boundaries retained as
metadata.

## Planning and checkpoint requirements

High-level search scores every prefix from one through `K1` and takes the
minimum. There is no remaining-budget input or mandatory stop latent. Planning
curves use `K0={8,16,32,64}` and `K1={1,2,4,8}`.

Training checkpoints contain both model and learner state. The learner state
includes EMA means and covariances, so planning reloads the trained
Mahalanobis geometry. Later-stage training requires its corresponding
counterfactual or value replay rather than accepting a stage label with no
stage-specific objective.

Every stage after the flat-token diagnostic requires both an earlier
checkpoint and a machine-readable passed admission record containing measured
metrics. Evaluation-only stages are rejected by the trainer.

For the staged baseline, checkpoint inheritance remains exact: sentence JEPA
loads token JEPA, commutation loads sentence JEPA, macro-action training loads
commutation, value distillation loads macro-action training, and closed-loop
reanalysis loads value distillation. The primary jointly trained token/sentence
model is the explicit exception: it starts from scratch, is labelled as joint
training in its checkpoint, and becomes the sentence checkpoint consumed by
subsequent worker and commutation gates. Intervening evaluation admissions bind
those exact training checkpoints rather than becoming checkpoint-producing
trainer stages.

Feature fingerprints cover tokenized problem and solution contents,
boundaries, symbolic metadata, frozen hidden states, model execution version,
and shard identity—not merely problem IDs. Counterfactual, candidate, value,
planner-replay, and admission artifacts bind that content fingerprint and the
SHA-256 of the exact source checkpoint.

## Compute accounting

Collector, replay, dense learner, value learner, and planner artifacts report
component wall time and estimated FLOPs. Text-only Qwen accounting excludes
the unused vision encoder and records whether optimized linear-attention
kernels or the PyTorch fallback were active. The common convention is two FLOPs
per frozen-model parameter-token for inference and six per trainable
parameter-item for forward/backward. Reports include each component's fraction
of total recorded FLOPs and wall time. CEM additionally records prior,
transition, objective, grounding, and whole-iteration latency separately.
The estimates are analytic monitoring quantities, not hardware-counter FLOPs.
