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

The v1 parser failures, v2 mixed-precision failures, and smaller-capacity v3
no-action cell are excluded from the scientific comparison. Action-only and
matched-action diagnostics remain required before upper-layer adaptation.
