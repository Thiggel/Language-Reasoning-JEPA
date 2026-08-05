# Training objectives

## Validated objective

| Objective | Weight | Gradient target |
|---|---:|---|
| Factual one-step latent prediction | 1.0 | encoders, predictor |
| Counterfactual one-step latent prediction | 1.0 | encoders, predictor |
| Endpoint logistic ranking | 1.0 | endpoint head, predictor, encoders |
| Root pair-difference distillation | 0.25 | Energy path and representation |
| Chunk prediction | 2.0 | predicted consequence content |
| VICReg | 1.0 | non-collapse |
| Dense rollout latent loss | 0 | — |
| Local GAR ranking | 0 | — |
| Absolute Endpoint-Energy MSE | 0 | — |

## Training scale

| Setting | Value |
|---|---:|
| Fresh examples per epoch | 30,000 |
| Epochs | 10 |
| Total examples | 300,000 |
| Batch size | 32 |
| Optimizer updates | about 9,370 |
| Learning rate | 3e-4 |
| Warmup | 500 updates |
| Gradient clip | 5 |
| Dropout | 0 |
| EMA momentum | about 0.99 → 0.999 |

## Gradient route

```text
endpoint ranking
 -> endpoint head
 -> recursively imagined endpoint
 -> every predictor application
 -> state and action encoders

EMA endpoint and goal labels
 -> stop gradient
```

## Required objective ablations

| Ablation | Question |
|---|---|
| No endpoint ranking | Is prediction alone sufficient? |
| No counterfactual prediction | Does alternative transition grounding help? |
| Counterfactual prediction only | Is ranking more than additional data? |
| No pair-difference auxiliary | Is the 0.25 term necessary? |
| Auxiliary weights `0.1,0.5,1` | Can one-step behavior improve? |
| Dense rollout loss | Does intermediate latent matching hurt? |
| Direct endpoint ranker | Is successor prediction necessary? |
