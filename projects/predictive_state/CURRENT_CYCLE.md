# Current cycle

`2026-08-10-qwen-frozen-transition-diagnostic-v1`

Decision: does the full state-plus-action predictor beat the no-action control
on held-out transition loss without an indexing or scale failure?

Direction-changing outcomes:

- Full beats no-action and action permutation while remaining rank-safe:
  proceed to the matched Qwen upper-half LoRA screen.
- Full matches no-action: audit action use and token/position alignment before
  training the backbone.
- Action-only matches full: the target is dominated by token identity; change
  target depth or add matched-action sampling before scaling.
- Non-finite loss, wrong layer shapes, or implausible target norms: treat as an
  implementation failure, not a negative scientific result.
