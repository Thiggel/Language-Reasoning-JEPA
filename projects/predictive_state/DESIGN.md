# Action-conditioned cross-layer prediction

## Thesis and scope

The project adds a small transition model to a causal LM. After processing
token `x_t`, it receives higher-layer residual states at position `t` and the
realized action `a_t = x_(t+1)`, then predicts the lower-layer residual that a
full teacher-forced transformer computes at position `t+1`:

`predicted h_(t+1)^m = G(h_t^u1, h_t^u2, E[a_t])`, with `m < u1 < u2`.

The main OLMo 2 1B split is `12,16 -> 8`; the cheap Qwen2.5-0.5B split is
`18,24 -> 12`. The scientific sequence is:

`predictive sufficiency -> stable recurrent state -> goal-directed geometry`.

Observed language modeling, recurrent execution, and reasoning geometry are
separate claims. Stage 3 oracle terminal states and candidate outcomes are
labeled and never silently promoted into an inference method.

## Stage 1: representation shaping

Define the source state

`s_t = [RMSNorm0(h_t^u1); RMSNorm0(h_t^u2)]`

and stationary target

`z_(t+1) = stop_gradient(RMSNorm0(h_(t+1)^m))`.

`RMSNorm0` has no affine parameters. The predictor emits a raw residual
`predicted h_(t+1)^m`; its normalized direction is compared with `z_(t+1)`.
The loss is

`L1 = L_NTP + λ_pred L_cos + λ_scale L_scale`,

where

- `L_cos = 1 - cosine(RMSNorm0(predicted h), z)`;
- `L_scale = Huber(log RMS(predicted h) - log RMS(target h))`;
- initial `λ_pred = 0.1` and `λ_scale = 0.01`;
- follow-up `λ_pred` values are `0.03, 0.1, 0.3`.

For squared prediction, the irreducible error is the expected conditional
variance `E tr Cov(z_(t+1) | s_t, a_t)`. Supplying the action removes
between-action variance. A full-versus-no-action advantage is meaningful only
when action-only, action permutation, and matched-action controls rule out
token identity as the complete explanation.

### Predictor

Each available source and the detached action embedding is projected to
`d/2`, concatenated, and passed through one bias-free SwiGLU block of width
`2d`. A linear skip maps the concatenated input directly to `d`. The output
projection is initialized with standard deviation `1e-3`; all other predictor
matrices use `0.02`; dropout is zero.

For `d = 2048`, the literal architecture contains about 46M weights, not
30–40M. Parameter count is therefore recorded from the instantiated model and
capacity-matched controls use the observed value rather than the estimate.

### Backbone adaptation

Stage 1 freezes embeddings and layers 1–8 for OLMo, making the target fixed.
Rank-16 LoRA with alpha 32 and zero dropout is inserted in Q/K/V/O and
gate/up/down projections of layers 9–16. The tied embedding/head remains
frozen. Qwen uses the fraction-matched lower 12/upper 12 split.

Optimization uses AdamW, predictor LR `3e-4`, LoRA LR `1e-4`, 2% warmup,
cosine decay to 10%, clipping at 1.0, BF16 where supported, predictor weight
decay 0.1, and LoRA weight decay zero. V100 diagnostics use FP16 because V100
does not support BF16.

### Budgets and controls

- Frozen Qwen diagnostic before joint training.
- Qwen2.5-0.5B screen: 20M tokens, length 1024, one seed.
- OLMo 2 1B main: 100M tokens, length 1024, global batch 65,536 tokens, three
  seeds after the screen selects one setting.

Required identical-token controls are NTP-only LoRA, no-action, action-only,
same-layer dynamics, NITP-style projection, frozen LM plus predictor, and an
equal-FLOP NTP run. The equal-FLOP control trains longer according to measured
step time; it is not estimated from parameter count alone.

### Predictor-removed evaluation

The normal full transformer is evaluated with no predictor call. Report
in-domain and OOD NLL, and zero-shot HellaSwag, PIQA, ARC, WinoGrande, and
LAMBADA when the evaluation harness is installed. Compare both equal tokens
and equal measured FLOPs.

Fresh, identically initialized probes are fitted after freezing each
checkpoint:

- conditional sufficiency at horizons 1, 2, 4, and 8 with ground-truth
  intervening actions;
- current-state versus an eight-state history window, with
  `Δ_history = error(Q1) - error(Q8)`;
- matched-action Spearman correlation between current-state distance and
  post-action target distance;
- future-passage retrieval among 31 negatives at offsets 32, 128, and 512;
- effective rank, mean pair cosine, covariance spectrum, centered-kernel
  alignment to the original checkpoint, and norm distribution.

Stage 1 passes only if transition error is lower than NTP-only, `Δ_history`
falls by at least 20%, rank does not collapse, and the predictor-removed model
improves NLL or downstream performance reliably.

## Stage 2: recurrent execution

After exact prompt prefill, action `a_t` produces `predicted h_(t+1)^8`.
Layers 9–16 consume that residual with the existing upper KV cache, exposing
predicted `h^12`, `h^16`, and next-token logits. This repeats without running
layers 1–8. Position IDs are tracked explicitly because lower-layer caches no
longer grow.

Starting from one exact state, recursively unroll ground-truth actions. At
step `k`, optimize direction error at layer 8, teacher-to-jump logit KL, and
`0.1` times jump CE. Weights are normalized `0.97^(k-1)`. Teacher states and
teacher probabilities are stopped; recurrent predicted states are not
detached.

The curriculum uses horizons 1, 4, 8, 16, and 32 for respectively 10M, 10M,
10M, 10M, and 20M tokens. Predictor LR is `1e-4`; LoRA LR is `5e-5`.
Beginning after horizon 16, 25% of batches use generated-prefix replay:

1. generate 64–128 actions with the jump model;
2. replay those exact tokens through the full model;
3. recover exact layer-8 targets on visited prefixes;
4. train recurrence on the frozen generated action sequence.

Evaluate teacher-action horizons 1 through 256 for normalized state error,
logit KL, excess NLL, top-1/top-20 agreement, norm drift, and perturbation
growth. Compare full decoding, jump-only decoding, refresh every 16/32/64
tokens, and jump drafting with exact speculative verification. Report NLL,
task accuracy, throughput, token latency, and measured cache bytes.

The practical gate is at least 90% task retention, less than 0.3 excess
nats/token at horizon 128, at least 1.4x throughput, and material KV savings.

## Stage 3A: explicitly oracle terminal geometry

Use an instruction-tuned checkpoint or reasoning SFT that retains the latent
objective. Start with GSM8K; reserve MATH-500 and another arithmetic dataset
for transfer. Generate at least four correct and four incorrect trajectories
where possible, retain states at newline/sentence boundaries, and require a
canonical final answer.

For a correct trajectory, the state immediately after its answer is the oracle
goal. This state is retrospective and unavailable during inference. Compare
raw cosine, raw Euclidean, train-whitened Euclidean, learned linear
Mahalanobis, and 128-dimensional train-fit PCA distances.

Report progress Spearman correlation, pairwise monotonicity over 1/2/4 chunks,
held-out remaining-chunk regression R², and matched-prefix correct-versus-
incorrect candidate ranking. Controls replace the goal with another correct
trajectory for the same problem, a different problem with the same numerical
answer, a random terminal, or the same relative position in another
trajectory. Compare original, Stage 1, and Stage 2 checkpoints.

## Stage 3B: usable learned distance and value

Predict a 128-dimensional goal `g_q` from the prompt state and project each
current state to `e_t`. Begin with `d(s_t,q) = ||e_t - g_q||₂`; separate state
and goal encoders permit some directionality. If it saturates, use a neural
quasimetric.

Chunk trajectories on newline boundaries. For correct trajectories train
terminal alignment, Huber regression to remaining chunks, and monotonicity
margin 0.25. Incorrect terminals receive distance margin 4. The total weights
are `1, 1, 0.5, 0.5`. Multiple correct trajectories for a problem share one
prompt-conditioned goal.

Train a separate direct value `P(success within budget b | s_t,q)` from
verified sampled-continuation outcomes. Compare distance value `exp(-d/τ)`,
direct value, and `logit(V_direct) - βd`. Distance must add information beyond
position, LM likelihood, entropy, and direct value.

## Exact-state gate and recurrent block beam search

Use batched block beam search before MCTS: beam width 4, four candidates per
beam, eight-token chunks, temperature 0.8, top-p 0.95, four-chunk lookahead,
and commit one chunk. Begin with 32-token lookahead and scale only after the
scorer passes.

Candidate score is length-normalized LM log probability times 0.2, minus goal
distance, plus direct-value logit. Uncertainty starts at zero. Run likelihood,
distance-only, value-only, and hybrid ablations.

The mandatory order is raw distance with exact states, learned distance with
exact states, direct value with exact states, hybrid with exact states, then
the identical frozen scorers with predicted jump states. Equal-compute
baselines are greedy, best-of-N, self-consistency, complete-sample value
reranking, ordinary beam, and value/distance-guided exact-state beam.

## Dense reward shaping

With frozen potential `Φ(s,q) = -d(s,q)`, shape verifier reward as

`r'_t = r_t + λ_shape [γ Φ(s_(t+1),q) - Φ(s_t,q)]`.

Retain the exact terminal verifier, clip increments to `[-0.1,0.1]`, and keep
total absolute shaping return at most half the terminal reward. Compare to
terminal-only PPO with identical rollouts and monitor verified success for
distance-model exploitation. Also report the empirical distance Bellman
residual `|d(s,q) - [1 + min_a d(F(s,a),q)]|` over the same candidate support.

## Execution and provenance

Grünau cells run from `git archive` snapshots in
`runs/autonomy/_code/<full-commit>/`. Every cell writes a project-qualified
run directory with state and exit markers, source revision, resolved CLI,
environment, dataset/checkpoint fingerprints, metrics, and checkpoints.
Before every launch, `gruenau-gpus` must show both low memory and low
utilization. A free-looking device can still race; the worker records the
actual host/GPU and fails cleanly on OOM.

Alex, Lise, and Grete use their existing shared environments and
`scripts/slurm_predictive_state.sbatch`; source is archived to each independent
filesystem, `PYTHONPATH` points at the snapshot `src`, and multiprocessing
temporary files are job-specific.
