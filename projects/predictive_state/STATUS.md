# Status

Implementation and the first frozen-backbone diagnostic are complete as of
2026-08-10.

At identical 101,522 boundary-safe WikiText-2 training transitions and equal
8,830,976-parameter predictors, full state-plus-action prediction reached
held-out cosine loss 0.2092 versus 0.3250 for the capacity-matched no-action
control, a 35.6% relative reduction. Permuting actions at evaluation raised
the full predictor's loss to 0.4414. This is positive diagnostic evidence that
the realized token supplies conditional information beyond current state.

It is not a representation result: the backbone was frozen, normal-path NLL
was necessarily identical, and the jointly trained predictor was reused.
Activation scale is not ready for recurrence: predicted/target RMS was
1.171/0.543 for full, and scale loss was worse than no-action (0.301 versus
0.190). Run the action-only diagnostic and tune scale calibration before the
20M-token upper-LoRA screen.
