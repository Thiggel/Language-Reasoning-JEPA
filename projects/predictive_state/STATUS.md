# Status

All three experimental stages and their operational wrappers are implemented.
An adversarial four-pass audit completed on 2026-08-10; corrected frozen
full/action-only diagnostics are the next active cells.

At identical 101,522 boundary-safe WikiText-2 training transitions and equal
8,830,976-parameter predictors, full state-plus-action prediction reached
held-out cosine loss 0.2092 versus 0.3250 for the capacity-matched no-action
control, a 35.6% relative reduction. Permuting actions at evaluation raised
the full predictor's loss to 0.4414. This is positive diagnostic evidence that
the realized token supplies conditional information beyond current state.

It is not a representation result: the backbone was frozen, normal-path NLL
was necessarily identical, and the jointly trained predictor was reused.
The v3/v4 objective accidentally applied scale weight as
`λ_pred × λ_scale = 0.001`, rather than the specified independent 0.01. Their
directional comparison remains a useful diagnostic, but their scale metrics
are not protocol-faithful and cannot gate recurrence. The corrected v5 full
and equal-capacity action-only cells use independent coefficients, packed
document attention, and WikiText article units. Do not begin the 20M-token
upper-LoRA screen until v5 is interpreted.
