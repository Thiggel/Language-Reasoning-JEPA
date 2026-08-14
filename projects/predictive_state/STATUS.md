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

## Stage 3 — 3A in progress

4,800 verified GSM8K trajectories from stock Qwen2.5-0.5B-Instruct, 29.6%
accuracy. 70% of problems yield at least one correct and one incorrect
trajectory; 28% yield no correct one and admit no oracle goal.

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
