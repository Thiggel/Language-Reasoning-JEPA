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
was identical, and the jointly trained predictor was reused. It is also a very
small run — 200 steps, 101,976 tokens, 10.1 seconds of wall clock, at context
256 on WikiText-2 in FP16. Held-out cosine loss and scale loss were both still
falling monotonically at the final step (scale 0.0647 at step 100, 0.0454 at
step 200; predicted RMS 0.769 then 0.724 against a target of 0.544), so the
1.33 RMS ratio is the state of an unconverged optimizer, not an established
negative. The scale term was also carrying almost no weight: at
`λ_scale = 0.01` it contributed 0.00045 of a total loss near 3.16.

Two measurement caveats apply to the headline direction number. The reported
`predicted_rms`/`target_rms` are linear-space means over a heavy-tailed
distribution (target norm mean 17.5, standard deviation 45.1, maximum 1679)
while the objective is a log-space Huber, so the two disagree about what is
wrong. And the target's effective rank is 2.13 of 896 dimensions with a
top covariance eigenvalue of 2072.9 against 8.96 for the second, so unrelated
target pairs already sit at cosine 0.268; a cosine of 0.787 is measured against
that floor, not against zero.

`2026-08-12-qwen-stage1-lora-screen-v1` was the first round in which part of
the language model adapts: rank-16 LoRA in layers 13–24, five token-matched
cells at 20M tokens each, WikiText-103 at context 1024 in BF16. All completed.

The transition objective is healthy and the scale question is closed. Full
reaches held-out cosine loss 0.1061 against 0.2078 for action-only and 0.2527
for no-action, with a +0.357 action-permutation gap; predicted/target RMS is
1.009 at `λ_scale = 0.01` and 1.0006 at 0.1, confirming that the frozen
diagnostic's 1.33 ratio was undertraining rather than an objective conflict.

Stage 1 does not pass. Predictor-removed NLL is identical across all five arms
to within 7e-4 nats, so the auxiliary objective neither helps nor harms the
ordinary path at `λ_pred = 0.1`. The LoRA weights do diverge between the full
and NTP-only arms, so this is a real null and not a broken gradient path.

Open before any adaptation claim: a frozen-backbone cell at the same 20M tokens
(the cosine improvement over the frozen diagnostic confounds adaptation with
200x more optimization), a persistence baseline against the 0.258 unrelated-pair
cosine floor, and a `λ_pred` sweep. See `CURRENT_CYCLE.md`.
