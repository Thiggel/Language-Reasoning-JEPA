# Current cycle

`2026-08-10-qwen-frozen-transition-diagnostic`

Observed decision: full beat the equal-capacity no-action control on held-out
direction error and depended strongly on correct action identity. The remaining
decision before joint training is whether action-only can explain the gain and
whether raw activation scale can be calibrated without harming direction.

Direction-changing outcomes:

- Full beats action-only as well as no-action and action permutation, with
  acceptable scale: proceed to the matched Qwen upper-half LoRA screen.
- Full matches no-action: audit action use and token/position alignment before
  training the backbone.
- Action-only matches full: the target is dominated by token identity; change
  target depth or add matched-action sampling before scaling.
- Non-finite loss, wrong layer shapes, or implausible target norms: treat as an
  implementation failure, not a negative scientific result.

The v1 runs failed before model loading because whitespace-only WikiText lines
were parsed incorrectly. The v2 runs failed at the first optimizer update due
to FP16 trainable weights. Both are operational failures. In v3, the full cell
completed but the no-action predictor was smaller; only the full result and its
same-capacity action-permutation audit are retained. The v4 no-action cell is
the admissible equal-parameter control.
