# Cycle: non-symbolic token-edit hierarchy on faithful iGSM

Status: invalidated before optimization; superseded by the data and
counterfactual pilot below

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
optimizer step. A deeper audit then showed that this was not merely a short
position table: the data adapter split an already tokenized official solution
on a standalone period token, although official iGSM steps commonly end in
fused tokens such as `6.`. Multi-step solutions consequently collapsed into
one long chunk. All five v3 runs are implementation-invalid and support no
architecture comparison. Official nested step boundaries are now preserved;
the positional table remains a conservative 320-token guard.

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

## Superseding decision: actual data and counterfactual coverage

The user explicitly requested that transferable ideas from the sibling
projects be tested in parallel, with special attention to the actual training
distribution and the amount of counterfactual data. The smallest current
decision is therefore: **how many unique synthetic repair trajectories, and
how many exact alternative outcomes per visited buffer, are required before
architecture experiments are meaningful?**

The audited task is oracle denoising, not naturally observed editing. Official
iGSM provides the problem and clean solution text. We corrupt the clean text
with literal token insertions, deletions, and replacements, then train on the
exact inverse stack. Corruption insert/replace tokens are sampled from that
example's gold-solution token pool, so trajectory generation is
candidate-privileged. The pool is not exposed in the model batch. The terminal
clean buffer is an EMA prediction target and is terminal-privileged. The
configured `fresh_per_epoch` flag previously changed only sampling order:
the trajectory for a seed and index is deterministic, so three epochs repeat
the same unique trajectories.

A 128-example audit after the boundary repair found exact terminal recovery
1.0 and multi-step collapse 0.0. Solutions contain a median 166.5 tokens and
9 official steps; the p95 maximum step is 43 tokens. Trajectories contain a
mean 11.14 edits, with delete/insert/replace fractions .341/.336/.323. Their
length is close to minimal token edit distance (mean ratio 1.029, maximum
1.25).

Counterfactual supervision now consists only of an action sampled from the
observed current buffer and its exactly executed next buffer. It has no
target-relative quality, defect, preference, or remaining-distance label.
This transfers the sibling projects' useful action-contrast idea without
transferring their candidate-privileged ranking signal.

The first round uses one seed and six cells: a 128-example H4 process smoke;
flat K=0 at 512, 2,000, and 6,000 unique trajectories for three passes; and
K=1 and 4 at the common 2,000-trajectory anchor. All K cells use the same
uniform-local alternative sampler. K=8 is implemented as a saturation
follow-up but was removed when the controller rejected the seven-cell design
at 18.92 GPU-hours against the sequence-edit 16-hour round limit. The admitted
six-cell round reserves 13.92 GPU-hours. This round deliberately precedes hierarchy,
dense-rollout, LDAD-removal, counterfactual-weight, repair-length, and
matched-update diversity comparisons.

Continue with the smallest K whose held-out exact-counterfactual or recursive
error is within 2% of the best, provided it improves at least 5% relative to
K=0 and worsens factual one-step error and effective rank by no more than 2%.
If no K passes, drop counterfactual outcome prediction. Select the data anchor
from the earliest clear saturation point before testing architecture.
