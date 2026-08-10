# Status

All three experimental stages and their operational wrappers are implemented.
An adversarial four-pass audit completed on 2026-08-10. Corrected frozen
full/action-only diagnostics then completed from revision `b051c23`.

The earlier v3/v4 comparison used identical 101,522 WikiText-2 transitions and
equal 8,830,976-parameter full/no-action predictors. Full reached held-out
cosine loss 0.2092 versus 0.3250 for no-action; action permutation raised full
loss to 0.4414. This is useful directional evidence, but those snapshots
accidentally applied scale weight as `λ_pred × λ_scale = 0.001`, so their scale
metrics are not protocol-faithful.

The corrected v5 full and equal-capacity action-only cells used independent
coefficients, packed document attention, and WikiText article units. At
identical token tensors and 101,976 admitted transitions, full reached cosine
loss 0.2133 versus 0.2910 for action-only, a 26.7% relative reduction. Correct
state therefore contributes beyond token identity. Permuting full-model actions
raised loss to 0.4497, so correct action identity also contributes.

This is not a representation result: the backbone was frozen, normal-path NLL
was identical, and the jointly trained predictor was reused. Scale improved
substantially under the corrected coefficient but is not ready for recurrence:
full predicted/target RMS is 0.724/0.544 (ratio 1.33), with scale loss 0.0454.
Run the smallest frozen scale-calibration cross-check before the 20M-token
upper-LoRA screen.
