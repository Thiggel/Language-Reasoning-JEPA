# Current cycle

First bounded token-oracle mechanism experiment for
[`REVISED_PLAN.md`](REVISED_PLAN.md).

Next falsifiable decision: frozen-LM proposal coverage and token-JEPA
flat oracle endpoint planning with exact candidate re-encoding. The frozen
checkpoint, tokenizer IDs, prompt, delimiter, counterfactual policy, and
symbolic-depth buckets are pinned in
[`NORMATIVE_CONTRACT.md`](NORMATIVE_CONTRACT.md).

Completed:

- smoke round `2026-07-30-nested-language-pilot-smoke-recovery-v5`;
- valid seed-0 round
  `2026-07-30-nested-language-first-five-paired-recovery-v3`;
- ID, near/far length-OOD, structural-OOD, and paraphrase-OOD curves over
  token horizons 4, 8, and 16;
- dense, counterfactual, and sparse-multistep comparisons at horizon 8;
- per-component FLOP and wall-time accounting.

The first scale attempt is retained as an excluded process failure: recursive
replay batch 32 exceeded 80 GiB. Recovery microbatch 4 completed without
changing data, losses, or effective examples.

The corrected Alex seed is pending as Slurm job `3928389`. Until it finishes,
all observations are single-seed and have wide uncertainty (eight roots per
cell). Do not advance to sentence JEPA from this result. The next decision is
whether replication supports another token-level coverage/model-error
experiment or rejects the current geometry/dynamics recipe.

The prior cycle,
`research/cycles/hard_text/2026-07-16-hierarchy-gradient-abstraction.md`, is
legacy.
