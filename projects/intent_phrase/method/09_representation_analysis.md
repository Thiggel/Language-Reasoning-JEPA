# Representation analysis

## Endpoint-ordering claim

Primary measurements at every rollout depth:

- pairwise endpoint-ranking accuracy;
- top-1 root-action recall;
- mean root regret;
- Spearman or Kendall correlation with exact continuation quality;
- Energy margin between best and second-best roots;
- correlation of each metric with planning success.

Compare against latent MSE, rollout error, effective rank, and probe scores.

## Language controls

| Pair | Expected relation |
|---|---|
| Same state, paraphrased text | similar state and action ranking |
| Same action, paraphrased intent | similar predicted endpoint |
| Lexically similar negation | different consequence |
| Operator swap | different endpoint |
| Irrelevant entity rename | preserved ranking |
| Same step, different remaining work | different progress |

Use full-space retrieval and distances as evidence. PCA, t-SNE, and UMAP are
illustrations only.

## Rollout analysis

Measure at depths `1,2,4,8,16`:

- predicted-versus-true endpoint distance;
- endpoint Energy calibration;
- root ranking before and after rollout;
- off-manifold score from a discriminator or nearest-neighbor distance;
- error conditioned on invalid, distractor, and necessary actions.

## Frozen probes

Probe operation, resolved count, remaining steps, step index, numerical value,
dependency properties, and invalid-action status. Control remaining progress
for elapsed position.

Effective rank diagnoses collapse. It is not evidence of planning geometry.
