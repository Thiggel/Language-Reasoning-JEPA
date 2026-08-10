# Current cycle

`2026-08-10-qwen-frozen-transition-diagnostic-v5`

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

The subsequent implementation audit found a nested scale-weight bug in all
v1–v4 code snapshots. V3/v4 remain interpretable for transition direction but
not scale calibration. V5 is the first protocol-faithful scale run and adds the
missing equal-capacity action-only control.

V5 completed. Full beat action-only by 26.7% in held-out cosine loss and was
strongly harmed by action permutation, satisfying the frozen-interface
direction test. Full activation RMS remains 33% above target. The next
falsifiable decision is a narrow scale-coefficient/LR cross-check; proceed to
upper-LoRA representation shaping only if calibration improves RMS without
erasing the direction advantage.
