# Sequence-edit status

The faithful non-symbolic token-edit path now has a replicated deployment-feasible
closed-loop result. At a 256-candidate budget over 128 test episodes, current-buffer
GAR improves normalized edit distance for two edits on both replay checkpoints, but
additional edits drift: horizon 3 underperforms horizon 2 on both checkpoints and
its third selected edit has negative mean true advantage. Use horizon 2 as the
fixed operating point.

The replay-depth/positive-anchor round `2026-07-19-structured-edit-replay-depth-anchor-wave53`
is still active. Do not admit a competing follow-up until its remaining jobs are
terminal and the completed and failed cells have been audited.
