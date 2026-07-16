# Cycle: non-symbolic token-edit hierarchy on faithful iGSM

Status: corrected relaunch blocked only by rolling GPU-hour limit

## Decision

Test whether temporal hierarchy improves predictive dynamics for literal token
edits over full official iGSM solution text.

## Interface and validity

The official hard generator supplies prompt and rendered solution text
(`max_op=21`, `max_edge=28`). Generic token insertion, deletion, and
replacement corrupt that text. The repair trajectory is the exact inverse of
those text operations. It never reads graph ancestry, feasibility, necessary
steps, defect counts, or symbolic action ranks.

The mutable buffer is encoded sentence-by-sentence into a shared state. A
causal primitive predictor consumes individual observed token edits. A causal
upper predictor consumes ordered chunks of four or eight edit embeddings and
predicts the corresponding future buffer state.

## Falsifiable pilot

One seed, 6k training examples, three epochs:

1. flat grounded token editor;
2. four-edit hierarchy;
3. eight-edit hierarchy;
4. four-edit hierarchy plus densely supervised primitive and macro rollouts;
5. four-edit hierarchy without faithful displacement-to-action decoding.

The first 20 GPU-hour envelope was rejected by the controller because it
would exceed the rolling budget. A 17.5-hour revision met the displayed but
not the fractional live limit. The unchanged five-cell matrix is therefore
capped at 205 minutes per cell, or 17.08 projected GPU-hours.

Launch v3 exposed a full-scale-only positional-cap bug before its first
optimizer step: one official rendered sentence was 276 GPT-2 tokens but the
encoder inherited a 128-token table. The cap is now explicit at 320, guarded
by a regression test, and an end-to-end faithful H4 training smoke passes.
The corrected v4 plan is validated and finalized, but the controller reports
119.58 GPU-hours already reserved in the seven-day window; launching the
17.08-hour matrix therefore requires an explicit budget increase.

All cells retain EMA targets, online-state VICReg, the frozen continuous
outcome target, and geometry-to-value self-distillation. Symbolic ranking,
scalar graph progress, monotonicity labels, and feasibility supervision are
zero.

Primary metrics are direct and recursive LN-L1 prediction error. Secondary
metrics are macro prediction error, state variance/effective rank, terminal
geometry correlation with remaining text edits, remaining-edit probe R2,
operation recovery from displacement, and raw action-token LDAD accuracy.

## Decision rule

- Continue hierarchy if H4 or H8 reduces recursive error and improves at least
  one progress/geometry diagnostic without lowering effective rank.
- Retain dense rollout only if it reduces recursive error at both levels.
- Retain LDAD only if its removal worsens transition or geometry diagnostics;
  action recovery alone is insufficient.
- Do not run paper seeds or claim planning until a non-oracle token proposal
  and closed-loop edit evaluator are added.

No unread sequence-edit steering note was present. The user explicitly
approved the faithful-iGSM transfer and prohibited symbolic supervision.
