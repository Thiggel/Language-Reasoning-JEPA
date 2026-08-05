# Training objectives and gradient routes

## Default Energy experiment

The August Energy experiments use a flat MLP JEPA with the following active
objectives:

| Objective | Default weight | Main purpose |
|---|---:|---|
| One-step latent prediction | 1.0 | Predict the EMA successor state |
| Counterfactual state prediction | 1.0 | Predict alternative feasible successors |
| GAR pairwise ranking | 1.0 | Order candidates by geometric continuation quality |
| Absolute Energy MSE | 0.25 | Calibrate Energy scale |
| Chunk prediction | 2.0 | Preserve local outcome content in predicted states |
| VICReg | 1.0 | Prevent collapse and maintain representation variance |

Other inherited objectives are configured with weight zero for this experiment.

## One-step latent prediction

For factual action `a_t`:

```text
z_hat_(t+1) = F_online(z_t, u(a_t))
z_target    = Encoder_EMA(h_(t+1))
```

The loss compares layer-normalized vectors with Smooth L1 distance. The EMA
target is detached.

## Counterfactual state prediction

At an anchor state, alternative feasible actions are executed by the training
environment to construct true one-step outcome histories. Those histories are
encoded by the EMA state encoder.

```text
counterfactual_prediction_i = F_online(z_t, u(a_i))
counterfactual_target_i     = Encoder_EMA(true history after a_i)
```

This objective gives alternatives latent-prediction supervision, not merely
ranking supervision. It is why a “counterfactual data without GAR” control is
required.

## GAR ranking

GAR consumes candidate Energies and geometric labels as detailed in
[03_gar.md](03_gar.md). With predicted-state input and `value_detach=false`,
ranking gradients reach the transition Energy head, predictor, action encoder,
and state encoder.

## Absolute Energy regression

The regression term calibrates each candidate independently:

```text
mean_valid (predicted_energy - teacher_target)^2
```

This is not the older pair-difference regression objective. The current
`GeoEnergyRegression` uses absolute targets. Its weight is swept because too
much regression can sacrifice ordering while too little can leave scores
incomparable during planning.

## Chunk prediction

The JEPA state predictor is also projected toward a target representation of
the next outcome chunk. This encourages predicted states to retain local
consequence content rather than only broad progress information.

The target chunk representation is obtained from a frozen anchor encoder in
the current MLP recipe. This choice is historical and should be stated when
comparing with causal or fully EMA-targeted variants.

## VICReg stabilization

VICReg is a variance-invariance-covariance regularizer. In ordinary terms it:

- keeps corresponding predictions and targets aligned;
- prevents every sample from receiving the same representation;
- discourages redundant dimensions from becoming perfectly correlated.

Effective rank and per-feature standard deviation are monitored as diagnostics,
but they do not establish planning geometry.

## EMA update

After each optimizer step:

```text
teacher_parameter
  <- momentum * teacher_parameter
     + (1 - momentum) * online_parameter
```

Momentum increases from approximately 0.99 toward 0.999 over training. The
teacher is updated only after the online optimizer step and never by gradient.

## Training scale

Current Energy screen:

- 30,000 fresh examples per epoch;
- 10 epochs;
- 300,000 generated training examples total;
- batch size 32;
- 9,370 optimizer updates because incomplete final batches are dropped;
- 500 validation examples after each epoch;
- learning rate 3e-4 by default, with a 7e-4 cross-check;
- cosine schedule with 500 warmup steps;
- dropout zero;
- gradient clipping at norm 5.

## Default gradient map

```text
latent prediction
  -> predictor and online encoders

counterfactual latent prediction
  -> predictor and online encoders

GAR ranking
  -> Energy head -> predicted successor -> predictor and online encoders

Energy MSE
  -> Energy head -> predicted successor -> predictor and online encoders

VICReg and chunk prediction
  -> representation and prediction networks

EMA targets
  -> no gradient
```

## Scientific controls for gradient routing

- **True-state Energy input:** removes the Energy-to-predictor gradient path.
- **Detached geometry:** should train a scorer from fixed geometric targets
  while blocking GAR geometry shaping.
- **Geometry only:** shape successor distances, then plan from raw distance
  without the amortized head.
- **Direct ranker:** remove successor prediction from the scoring path.
- **Counterfactual-only:** retain alternative transition supervision but remove
  ranking.

These controls answer different questions and should not be merged under a
single “no GAR” label.
