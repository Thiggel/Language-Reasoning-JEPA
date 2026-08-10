# Evidence

## Frozen Qwen diagnostic

Admitted process cells:

- `runs/autonomy/predictive_state/2026-08-10-qwen-frozen-transition-diagnostic-v3/qwen05-frozen-full-s0-v3/`
- `runs/autonomy/predictive_state/2026-08-10-qwen-frozen-transition-diagnostic-v4/qwen05-frozen-no-action-capacity-matched-s0-v4/`

Both used Qwen2.5-0.5B revision `060db649...`, 200 updates, 101,522
boundary-safe ordinary-language transitions, context 256, seed 0, and
8,830,976 predictor parameters.

| Metric | Full | No action |
|---|---:|---:|
| Held-out cosine loss | 0.2092 | 0.3250 |
| Normalized MSE | 0.4184 | 0.6500 |
| Scale loss | 0.3013 | 0.1902 |
| Predicted RMS | 1.1706 | 0.9939 |
| Target RMS | 0.5428 | 0.5428 |
| Predictor-removed NLL | 3.0684 | 3.0684 |

For the full predictor, held-out action permutation increased cosine loss to
0.4414, a +0.2322 loss gap. The full-versus-no-action direction reduction is
35.6% relative. This supports conditional action information in the frozen
state interface, but not representation improvement or recurrent readiness.
The scale mismatch is an explicit negative result for immediate injection.

Post-run audit found that these cells used an effective scale coefficient of
0.001 because `λ_scale` was incorrectly nested under `λ_pred`. Consequently,
the raw scale numbers diagnose that implementation only; they are not evidence
against the specified 0.01 scale objective. Directional loss and the action
permutation diagnostic remain descriptive frozen-interface evidence, not an
admitted Stage 1 result.

The v1 parser failures, v2 mixed-precision failures, and smaller-capacity v3
no-action cell are excluded from the scientific comparison. Action-only and
matched-action diagnostics remain required before upper-layer adaptation.

## Corrected audit cells

`2026-08-10-qwen-frozen-transition-diagnostic-v5` reran full and action-only
with equal 8,830,976-parameter predictors, independent objective weights,
fixed validation samples, article-level WikiText documents, and packed
block-diagonal attention. Both completed at 101,976 valid transitions from
source revision `b051c23`. All train/validation tensors were bit-identical;
their path-independent tensor digest is
`5290dc0206a3f1db11d8568625120bba19481f5cb83f07c4966c4533fb2e4b66`.

| Metric | Full | Action only |
|---|---:|---:|
| Held-out cosine loss | 0.2133 | 0.2910 |
| Normalized MSE | 0.4266 | 0.5821 |
| Scale loss | 0.0454 | 0.0139 |
| Predicted RMS | 0.7243 | 0.5844 |
| Target RMS | 0.5438 | 0.5438 |
| Permuted-action cosine loss | 0.4497 | 0.5405 |
| Predictor-removed NLL | 2.9746 | 2.9746 |

Full reduces direction error by 0.0777, or 26.7% relative to action-only. Its
action-permutation loss gap is +0.2364, and matched-action state/consequence
distance Spearman is 0.6625 over 2,836 pairs. This is positive frozen-interface
evidence that both state and realized action matter. It remains neither a
representation result nor a recurrent-decoding result. The full predictor's
1.33 RMS ratio requires calibration before injection.
