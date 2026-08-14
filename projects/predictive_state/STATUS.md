# Status

Current observed state only. Detail lives in the dated report
`research/reports/predictive_state/2026-08-14-stage1-closed-stage2-started/`.
What is running lives in `CURRENT_CYCLE.md`.

## Stage 1 — closed, small positive, gate not met

Fifteen token-matched cells and two independent probes. The objective makes the
representation slightly more transition-sufficient at essentially no cost, and
far less than the charter requires.

At 300M FineWeb-Edu tokens, full finetuning of layers 13-24, against a
re-measured NTP-only control: predictor-removed NLL 2.6235 versus 2.6206, a
+0.0029 nat cost. Fresh probes on the frozen checkpoints give H1 transition
error 0.2989 against the control's 0.3033, with the aux-leaning cell at 0.2946.
Consistent across horizons 1/2/4/8 and dose-responsive, but only 1.5-2.9%
relative. Language modeling never improves.

Ruled out as explanations, each by direct measurement: that no pressure reaches
the backbone (with NTP off it moves to NLL 5.69), that the prediction weight was
never raised enough (it was, but NTP's hard-coded weight of 1.0 kept the loss
8.2x NTP-dominated even at weight 3.0), and that the residual error is a
fixed-state compression limit (error is flat across positions 0-1023).

## Stage 2 — efficiency passes, fidelity does not

Untrained open-loop rollout: state error does not diverge over 256 recursive
steps (0.1436 at horizon 1, 0.1405 at 256). Measured throughput 1.86x against a
1.4x gate; decode cache growth exactly halves. Excess NLL saturates near 1.11
nats against a 0.3 gate, top-1 agreement near 0.52, top-20 near 0.90.

A first horizon curriculum improved long-horizon excess NLL about 9% to 1.01 and
worsened state error. It ran roughly 7M tokens against the protocol's 60M, so it
does not settle the question.

Jump drafting with exact speculative verification is the untested operating
point that the top-20 figure points at, and is lossless by construction.

## Stage 3A — complete, negative

4,800 verified GSM8K trajectories from stock Qwen2.5-0.5B-Instruct, 29.6%
accuracy, 251 trajectories with an oracle terminal.

The oracle goal geometry is generic progress, not goal direction. A learned
Mahalanobis metric reaches progress Spearman 0.7663 toward the true goal but
0.7257 toward a *random* terminal from another problem, so goal specificity is
0.041. Raw cosine is the same shape smaller: 0.2571 against 0.1929. A
mid-trajectory state at the same relative position scores only 0.2759, so it is
terminal-ness the metric tracks rather than position.

It cannot rank correct above incorrect. Matched-prefix accuracy is 0.375, 0.375,
0.450, 0.425 and 0.300 against 0.5 chance, worst for the learned metric. That is
the quantity planning needs. Reported as a negative and not built on, with the
caveat of a 0.5B model at 29.6% accuracy and only 40 matched-depth pairs.

## Continuous generation: refresh is flat in N

`2026-08-14-qwen-stage2-refresh-sweep-v1`, 256 actions over 8 prompts, block
refresh materializing only the new block. Agreement with full greedy, and
modelled speedup from measured per-step costs, on the Stage 2 checkpoint:

| setting | agreement | first divergence | speedup |
|---|---:|---:|---:|
| jump only | 0.153 | 7.9 | 1.86x |
| refresh 128 | 0.153 | 7.9 | 1.85x |
| refresh 32 | 0.151 | 7.9 | 1.81x |
| refresh 8 | 0.156 | 9.1 | 1.68x |
| refresh 4 | 0.164 | 9.5 | 1.53x |
| refresh 2 | 0.219 | 20.2 | 1.30x |

Refreshing more often barely helps until N reaches 2, where the speedup has
already fallen to 1.30x. This is the predicted signature of a per-step fidelity
limit rather than accumulated drift: state error was already flat to horizon
256, so there was little drift for a refresh to correct. The Stage 2 curriculum
did help autonomous generation, doubling agreement from 0.062 to 0.153 and
first divergence from 5.2 to 7.9 tokens.

Speculative decoding at draft length 2 reaches acceptance 0.957 and a modelled
1.38x, which is lossless. Its equality check needed fixing: a cached and an
uncached bf16 forward of the same model agree only 0.52 with each other, so the
verifier's uncached path must be compared against an uncached reference.

## Energy head: transfers to correctness, does not beat likelihood

`2026-08-14-qwen-energy-head-v1`. Trained only to rank the continuation the
dataset records above sampled alternatives, with no verifier and no step
counter: held-out ranking accuracy 0.815 against a likelihood baseline of 0.395,
so it is not re-deriving likelihood on its own task.

Transferred to 3,200 externally verified trajectories, with the verifier used
for evaluation only:

| | correctness AUC | matched-problem accuracy |
|---|---:|---:|
| energy head | 0.673 | 0.613 |
| LM likelihood | **0.726** | **0.635** |

An energy trained with no correctness signal does predict correctness well above
chance, which is the encouraging half. It does not clear the charter's gate,
which requires adding information beyond LM likelihood, and likelihood beats it
on both measures. The likely cause is provenance leakage: the negatives are
model samples and the positives are dataset gold, so the head can separate them
on style, and gold style correlates with correctness only indirectly. The fix
that keeps the objective self-supervised is to draw negatives from the same
distribution as the positives, for example gold steps from other problems, so
only contextual relevance differs.

## Architectural facts that outlived Stage 1

- The transition is close to linear: 0.1335 for a full-rank linear predictor
  against 0.1067 for the SwiGLU, and 0.2527 with no state.
- Its state contribution is low dimensional: eight state dimensions reach
  0.1508 against 0.2078 for action-only.
- Layer 24 is redundant: layer 18 alone reaches 0.1050, layer 24 alone 0.1252.
- Activation scale is solved: RMS ratio 1.0006 at `λ_scale = 0.1`.

## Owed

A frozen-backbone cell at matched tokens, a persistence baseline against the
0.258 unrelated-pair cosine floor, and a long `ntp_weight = 0` cell. Everything
is single seed on Qwen2.5-0.5B.
