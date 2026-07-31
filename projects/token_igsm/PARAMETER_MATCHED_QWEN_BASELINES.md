# Parameter-matched Qwen controls

The first scaled nested-language experiment uses the pinned non-thinking
`Qwen/Qwen3.5-0.8B` model and the same verified iGSM train split for every
condition.

The active token-JEPA stack has 77,329,408 trainable parameters:
`E_0`, the token-action embedding, and `P_0`. EMA targets are training
state—not deployable model capacity—and later sentence modules are not counted
in this Stage-1 comparison.

Two controls use exactly that scalar budget:

1. **Added-capacity Qwen.** Three newly initialized native Qwen
   linear-attention decoder blocks plus one residual SwiGLU layer add exactly
   77,329,408 parameters. Only these new parameters train. This control tests
   whether ordinary autoregressive capacity explains the JEPA result.
2. **Unfrozen Qwen.** Model capacity is unchanged. Exactly 77,329,408
   pretrained scalar weights in the top four decoder blocks receive gradients.
   Whole tensors are selected first and one deterministic partial tensor mask
   supplies the exact remainder. Inactive scalars in that tensor receive
   neither gradients nor weight decay.

Both controls use solution-only causal cross-entropy, excluding prompt,
padding, and terminal special tokens. They see the same deterministic
tokenized examples and the same 50,000 optimizer updates (batch size eight) as
the scaled JEPA run. The newly initialized control uses peak LR `1e-3`; the
pretrained unfreezing control uses `1e-5`. Both use 5% linear warmup followed
by cosine decay, and save checkpoints at 25%, 50%, 75%, and 100% of training.

The submitted campaign uses one seed and 25% of the reference split counts:
50,000 train problems plus held-out ID, length-OOD, structural-OOD, and
paraphrase-OOD subsets. Checkpoint curves diagnose undertraining; multi-seed
confirmation remains necessary for final small-effect claims.

Qwen final-answer accuracy is a deployable metric. Flat token-JEPA planning is
an explicitly **candidate-privileged oracle diagnostic** and must not be
presented as directly comparable task accuracy. Proposal coverage, exact
endpoint selection, learned-rollout selection, and final-answer generation are
reported in separate panels/tables.
