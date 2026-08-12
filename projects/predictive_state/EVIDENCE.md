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

## Stage 1 upper-half LoRA screen

`2026-08-12-qwen-stage1-lora-screen-v1`, revision `51a799b`. Five cells, each
19,964,512 tokens at context 1024, BF16, seed 0, from one shared token-block
file (tensor digest `1365dc54…`, WikiText-103 articles, 37,106 train and 778
validation blocks). Rank-16 LoRA in layers 13–24 (4,399,104 trainable weights);
embeddings and layers 1–12 frozen, so the layer-12 target is a fixed anchor.
Predictor 8,830,976 parameters. All arms consumed an identical batch order.

| Metric | full | no action | action only | NTP only | full, `λ_scale` 0.1 |
|---|---:|---:|---:|---:|---:|
| Predictor-removed NLL | 2.549868 | 2.549854 | 2.549918 | 2.549929 | 2.550545 |
| Held-out cosine loss | 0.1061 | 0.2527 | 0.2078 | — | 0.1067 |
| Permuted-action cosine loss | 0.4630 | — | 0.5569 | — | 0.4631 |
| Scale loss | 0.0035 | 0.0047 | 0.0032 | — | 0.0010 |
| Predicted / target RMS | 0.550/0.545 | 0.548/0.545 | 0.545/0.545 | — | 0.545/0.545 |

Two findings, one positive and one null.

The transition result strengthens once the source half can adapt. Full reduces
direction error 49% relative to action-only and 58% relative to no-action, with
an action-permutation gap of +0.357. Matched-action Spearman is 0.732. All
cosine curves are flat from step 1000, so these are converged values.

Activation scale is no longer a blocker, and the frozen diagnostic's 1.33 RMS
ratio was an artifact of stopping after 200 steps. At the protocol coefficient
the ratio is 1.009 with scale loss 0.0035; at `λ_scale = 0.1` it is 1.0006 with
scale loss 0.0010 and cosine unchanged at 0.1067. The larger coefficient is
free and should be the default.

The null is on language modeling. Predictor-removed NLL is the same across all
five arms to within 7e-4 nats, and full beats the token- and capacity-matched
NTP-only arm by 0.00006 nats. The LoRA weights of the two arms do differ (all
168 tensors, maximum elementwise difference 6.6e-4 against a maximum magnitude
of 4.1e-2), so the auxiliary gradient does reach the backbone; its effect on
the ordinary path is simply negligible at `λ_pred = 0.1`. Stage 1's gate asks
for a reliable NLL or downstream improvement, and this run does not supply one.

Target geometry is identical across arms, as the frozen anchor requires:
effective rank 2.977 of 896, mean pairwise cosine 0.2579. Because unrelated
target pairs already sit at 0.258, the absolute cosine values above are
measured against a high floor, and a persistence baseline is still owed.

This is not yet a Stage 1 pass, and the improvement over the frozen diagnostic
confounds backbone adaptation with 200x more optimization; a frozen-backbone
arm at the same 20M tokens is required to separate them.
