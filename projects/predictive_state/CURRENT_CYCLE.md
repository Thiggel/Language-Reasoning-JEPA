# Current cycle

`2026-08-12-qwen-stage1-lora-screen-v1`

The frozen diagnostic answered what it could. Because layers 1–24 were all
frozen there, nothing in the language model could change, so the predictor
could only chase a stationary target and predictor-removed NLL was a constant.
This cycle is the first in which any part of the model adapts.

Five token-matched cells share one pre-built token-block file
(`_data/wikitext103_qwen_ctx1024.pt`, 37,106 train and 778 validation blocks at
context 1024, tensor digest `1365dc54…`) and an identical batch order, so they
differ only in the auxiliary objective:

| Cell | Variant | `λ_scale` | Role |
|---|---|---:|---|
| `qwen05-lora-full-s0-v1` | full | 0.01 | treatment |
| `qwen05-lora-ntp-only-s0-v1` | ntp_only | — | identical LoRA capacity and tokens, no predictive loss |
| `qwen05-lora-no-action-s0-v1` | no_action | 0.01 | state only |
| `qwen05-lora-action-only-s0-v1` | action_only | 0.01 | action only |
| `qwen05-lora-full-scale0.1-s0-v1` | full | 0.1 | scale-coefficient probe |

Embeddings and layers 1–12 stay frozen, so the layer-12 prediction target is a
fixed anchor: only the source half (13–24, rank-16 LoRA, 4,399,104 trainable
weights) and the 8,830,976-parameter predictor adapt. Unfreezing the target
stack would let the target drift to meet the predictor, which is the collusion
failure already observed in the sibling subprojects.

Corpus changed from WikiText-2 to WikiText-103 articles. At 20M tokens
WikiText-2 would have been roughly nine epochs, which would have confounded any
held-out NLL movement with memorization; 20M tokens is now under half an epoch.
Precision is BF16 on H100, removing the V100 FP16 caveat.

All five cells completed on 2026-08-12 at revision `51a799b`; numbers are in
`EVIDENCE.md`. The observed outcome is the third one below crossed with the
second: the transition objective works well and costs nothing, but it also buys
nothing on the ordinary path. Full reaches held-out cosine loss 0.1061 against
0.2078 for action-only and 0.2527 for no-action, with an action-permutation gap
of +0.357, while predictor-removed NLL is identical to the NTP-only control to
within 7e-4 nats. Activation scale is solved: the RMS ratio is 1.009 at the
protocol coefficient and 1.0006 at `λ_scale = 0.1`, which costs nothing in
direction and should become the default.

Next, in order:

1. Frozen-backbone `full` at the same 20M tokens. The jump from cosine 0.2133
   to 0.1061 confounds backbone adaptation with 200x more optimization, and
   this one cell separates them. Until it runs, no adaptation claim is safe.
2. Persistence baseline on these checkpoints, per item 1 below.
3. `λ_pred` sweep at 0.03 and 0.3. The NLL null was measured at one coefficient
   only; 0.3 is the setting that could plausibly move the ordinary path, and
   0.03 bounds the tax if 0.3 hurts.

Direction-changing outcomes as pre-registered:

- Full improves predictor-removed held-out NLL over the NTP-only arm at equal
  tokens, without rank collapse: first genuine Stage 1 representation signal;
  proceed to the seed replication and then the OLMo main comparison.
- Full matches NTP-only on NLL but keeps its transition advantage: the
  objective shapes the transition without paying for it in language modeling.
  That is a weaker, still-publishable claim; it moves the emphasis to Stage 2.
- Full is worse than NTP-only on NLL: the auxiliary loss taxes the normal path
  at `λ_pred = 0.1`; sweep `λ_pred` down before anything else.
- The full-versus-action-only advantage seen under a frozen backbone shrinks
  once the sources adapt: the earlier gain depended on a fixed interface, and
  the target depth or the source split must be revisited.

Open items this cycle does not settle, in priority order:

1. Persistence baseline. Effective rank of the target is ~2.1–2.9 out of 896
   and unrelated target pairs already sit at cosine 0.26, so a cosine of 0.79
   is measured against a high floor. How well a trivial copy predicts
   `h_(t+1)^12` is the first control a reviewer will ask for. It needs no
   training and runs on the resulting checkpoints.
2. Outlier-robust scale reporting. `predicted_rms`/`target_rms` are linear
   means over a heavy-tailed distribution (norm mean 17.5, max 1679) while the
   objective is a log-space Huber, so the headline ratio and the optimized
   quantity are not the same number.
3. One seed only, as the screen protocol specifies.
