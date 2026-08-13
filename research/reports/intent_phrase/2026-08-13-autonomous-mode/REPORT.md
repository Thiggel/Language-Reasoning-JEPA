# Autonomous mode: the system writes its own solution, and where that breaks

_2026-08-13. The fully menu-free, executor-free interface: codebook proposals
(no menu) -> environment-side grounding -> endpoint-Energy planning over
imagined states -> the DETACHED frozen-state decoder emits the next step's
text -> the model re-encodes its own text and continues -> only the FINAL
ANSWER is scored. The oracle is used solely to compute ground truth.
Code: src/textjepa/planning/autonomous.py (1f5e644, gate in ca310d7);
`candidate_interface=autonomous`. Smoke scale: 15-25 stylized-iGSM val
problems, checkpoint stab-ldad-ema-s0-v1 + decoder-s0-v1, CPU, greedy and
seeded (an episode is a pure function of ckpt, decoder, problem, seed)._

## Result: final answers are at chance; the reason is arithmetic, not planning

| metric | gated (default) | ungated ablation |
|---|---|---|
| final-answer accuracy | 0.067 | 0.067 (0.040 at n=25) |
| emitted-value correct (per step) | 0.182 | 0.125 |
| well-formed / about the chosen action | 0.909 | 0.875 |
| mean steps (mean necessary 4.33) | 2.20 | 1.60 |
| completion-claim rate | 0.733 | 0.933 |
| stall rate | 0.267 | 0.000 |
| mean first divergence (1-based step) | 1.14 | 1.00 |
| gate pass rate | 0.626 | — |

Chance is 1/23 = .043. ~90% of self-generated sentences are well formed and
about the right variable; only 8-18% carry the right number, and the first
wrong number is at step 1. This is the 2026-08-11 state-decoder finding
(imagined states keep discourse structure, lose arithmetic) reproduced
inside a closed loop. Qualitative failure mode: emitted numbers get STUCK on
one value across steps — "... is 9 times 8 = 9 .", "... is 9 plus 0 = 9 .",
"... is 9 ."

## Second finding: without an executor nothing enforces prerequisite order

Ungated, the planner claims completion after ~1.6 steps on problems needing
4.3: the endpoint Energy rewards reaching the goal state and no environment
refuses an infeasible action, so it writes the queried variable's sentence
immediately. This is a structural property of executor-free operation, not a
bug — the environment had been silently supplying the feasibility gate.

## The gate: our own mechanism, applied

We already have a non-oracle feasibility detector — LDAD cycle-consistency
(AUC .94: feasible actions produce action-identifiable displacements, mean
token log-prob -0.28, infeasible -4.5). The gate thresholds this score
before the Energy search sees a candidate, at roots and beam expansions.
Threshold calibrated on TRAINING problems only (feasibility labels used in
calibration, never at plan time): midpoint between feasible/infeasible score
means = -4.22. A 10th-percentile variant was tried first and was nearly
vacuous (82% pass) because feasible scores have a long low tail.

Effect: rejects ~37% of proposals; shortcutting is REDUCED, not eliminated
(steps 1.6 -> 2.2, immediate claims .93 -> .73, and a quarter of episodes now
stall rather than write a bogus answer). AUC .94 leaves enough overlap that
one scalar threshold cannot enforce a dependency order. Final-answer accuracy
does not move.

## Reading for the paper

The gate improves the PROCESS (fewer illegal shortcuts, fewer fabricated
answers) without moving the OUTCOME, because the outcome is bounded by
single-step predictor fidelity on values. Autonomous mode therefore supports
the preregistered claim — *the planner picks better plans; rendering the
computation is the bottleneck* — and additionally isolates a second
requirement for executor-free reasoning: a feasibility/termination
discipline stronger than a scalar threshold on a .94-AUC score.

Limitations: single checkpoint/decoder/seed, 15-25 episodes, CPU smoke,
stylized iGSM only (faithful needs its own state decoder and step parser;
plan.py raises rather than silently falling back to a menu). These numbers
are a smoke measurement, not a paper row — the paper row comes from the
5-seed budget checkpoints.
