# Action-conditioned predictive state: Stage 1 closed, Stage 2 opened

2026-08-12 to 2026-08-14. Qwen2.5-0.5B, split `18,24 -> 12`.

## What the subproject is testing

A small transition model reads two upper residual states at position `t` plus
the token that actually came next, and predicts the residual that the frozen
lower stack computes at position `t+1`:

`G(h_t^18, h_t^24, E[a_t]) -> h_(t+1)^12`

Layers 1-12 and the embeddings stay frozen in every cell, so the target is a
fixed anchor and a collapsing source representation raises the loss rather than
lowering it. Three separable claims: the objective improves the ordinary model
(Stage 1), the predicted state can drive decoding without the lower half
(Stage 2), and the resulting geometry supports planning (Stage 3).

## Stage 1: the objective is nearly free, and nearly useless

Fifteen token-matched cells across three rounds, plus two independent probes.

**It does not shape the representation in any way that helps language
modeling.** On WikiText-103 at 20M tokens, nine configurations spanning a 30x
prediction-weight range, a 56x state bottleneck, a linear-only predictor and
single-source conditioning all landed within 0.0021 nats of the token- and
capacity-matched NTP-only control. The decisive cell narrowed the predictor's
state channel to 8 dimensions, which more than doubled its direction error from
0.1067 to 0.2401; the backbone did not compensate, it simply accepted the worse
transition loss.

**Three of my own explanations for that were wrong and were retracted.**

- "No pressure reaches the backbone." False. With the next-token loss switched
  off entirely the backbone moved enormously, to NLL 5.69 from 2.55.
- "The pressure round tested the objective's strength." False, and this one
  mattered. The next-token term carried a hard-coded weight of 1.0, so even at
  prediction weight 3.0 the measured loss was still 8.2x next-token dominated
  (2.469 against 0.301). The sweep went from 80x dominance to 8x dominance and
  never reached the regime it claimed to test. An explicit `ntp_weight` was
  added.
- "The residual error is a fixed-state compression limit." False. If a
  fixed-size state were failing to summarize a growing prefix, error would climb
  with position. It is flat: last position bucket over first is 1.00, 0.98 and
  0.97 across three checkpoints over positions 0-1023.

**At continual-pretraining scale the cost disappears but the benefit stays
small.** Moving to FineWeb-Edu removed a confound: WikiText-103 is a domain
shift for a web-pretrained model, so part of the earlier NLL penalty was domain
adaptation rather than the objective. At 300M tokens, global batch 65,536, full
finetuning of layers 13-24:

| Cell | NLL | vs control | co-trained cosine |
|---|---:|---:|---:|
| NTP-only control | 2.6206 | — | — |
| prediction weight 3, NTP weight 1 | 2.6235 | +0.0029 | 0.0920 |
| prediction weight 1, NTP weight 0.1 | 2.6414 | +0.0208 | 0.0868 |

Fresh probes fitted on the frozen checkpoints, which is the measurement the
protocol requires because the co-trained predictor measures system fit rather
than representation quality:

| fresh probe on | H1 | H2 | H4 | H8 |
|---|---:|---:|---:|---:|
| NTP-only control | 0.3033 | 0.3167 | 0.3104 | 0.3224 |
| prediction weight 3 | 0.2989 | 0.3142 | 0.3105 | 0.3215 |
| prediction weight 1, NTP 0.1 | 0.2946 | 0.3121 | 0.3102 | 0.3212 |

So the objective does make the representation more transition-sufficient, the
effect is consistent across horizons and dose-responsive, and it costs 0.003
nats. But it is 1.5-2.9% relative, far short of the charter's "substantially
more transition-sufficient". The `Δ_history` gate is technically met by the
aux-leaning cell (0.0036 to 0.0017, a 53% fall against a 20% requirement) but
should not be leaned on: the gap is ~0.003 on a quantity near 0.30, so an
eight-state history window barely helps any checkpoint. The gate was written
assuming that gap would be large.

Caveat on the fresh probes: 800 steps only, landing near 0.30 against the
co-trained predictor's 0.092. They are undertrained, which compresses
differences between checkpoints. Only the differences are meaningful.

**Verdict.** Stage 1's representation claim is a small, nearly free, real
effect that does not meet its gate. It should be reported as such.

## Architectural findings that outlived Stage 1

- The transition is close to linear. A full-rank linear predictor reaches
  cosine 0.1335 against 0.1067 for the SwiGLU, and 0.2527 with no state at all,
  so the nonlinearity buys about a fifth of what the state buys.
- The state contribution is low dimensional. Eight state dimensions with the
  action channel at full width reach 0.1508 against 0.2078 for action-only.
- Layer 24 is redundant. Layer 18 alone reaches 0.1050, marginally better than
  both sources together; layer 24 alone reaches 0.1252.
- Activation scale is solved. The frozen diagnostic's 1.33 RMS ratio was
  undertraining, not an objective conflict: at `λ_scale = 0.1` the ratio is
  1.0006 and cosine is unchanged.

## Stage 2: efficiency passes, fidelity does not

Open-loop rollout on the Stage 1 checkpoints, before any Stage 2 training:

| horizon | state error | excess NLL | top-1 | top-20 |
|---:|---:|---:|---:|---:|
| 1 | 0.1436 | 0.333 | 0.63 | 1.00 |
| 16 | 0.1532 | 1.060 | 0.47 | 0.89 |
| 256 | 0.1405 | 1.114 | 0.49 | 0.90 |

State error does not diverge over 256 recursive steps, which was the main risk.
Measured throughput is 1.86x against a 1.4x gate and decode cache growth halves
exactly as predicted. What fails is fidelity: excess NLL saturates near 1.11
nats against a 0.3 gate.

A first horizon curriculum (1, 4, 8, 16, 32 with generated-prefix replay from
16) improved long-horizon excess NLL by about 9%, to 1.01, and made state error
worse. It was under-budgeted at roughly 7M tokens against the protocol's 60M,
so it does not settle the question, but 9% for 7M tokens does not extrapolate
to a 70% reduction.

**The operating point that sidesteps the gate.** Top-20 agreement holds near
0.90 while top-1 is near 0.52. That is the profile for jump drafting with exact
speculative verification, which the design already lists. Verification makes it
lossless by construction, so the excess-NLL gate does not apply at all, and a
52% acceptance rate with 1.86x cheaper drafting is still a real speedup. This is
probably the defensible form of the Stage 2 claim and it needs no further
training to evaluate.

## Stage 3A: started

4,800 GSM8K trajectories from stock Qwen2.5-0.5B-Instruct (revision `7ae5576`),
16 samples over 300 problems, 29.6% verified accuracy. 70% of problems yield at
least one correct and one incorrect trajectory; 31% reach the design's target of
four and four; 28% have no correct trajectory and so admit no oracle goal.
Correctness comes from exact numeric agreement with the dataset gold answer and
is recorded as candidate-privileged. State collection and the oracle geometry
probe are running.

## Normative conflict to resolve before Stage 3B

The 2026-08-14 rule that steps-to-go regression and symbolic ranking labels are
out of scope, even as diagnostics-turned-components, conflicts with Stage 3B as
written in `DESIGN.md`, which trains the goal distance with "Huber regression to
remaining chunks" and a margin driven by verified correct/incorrect terminals.

Stage 3A is unaffected: it is a labeled oracle measurement, not a component, and
it answers whether the geometry exists at all. Stage 3B needs rewriting so the
energy head is trained by ranking counterfactual continuations against the one
that actually occurred, which is self-supervised, with the verifier kept for
evaluation only.
