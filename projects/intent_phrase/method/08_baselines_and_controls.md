# Baselines and controls

## Headline baselines

| Model | Selection rule |
|---|---|
| Random | random candidate |
| Token LM | normalized intent-token likelihood |
| Sentence LM | next-intent likelihood |
| Sentence LM plus latent loss | generative and predictive hybrid |
| Latent-MSE JEPA | predicted consequences without Endpoint Energy |
| Direct endpoint ranker | scores action sequence without predicted states |
| Endpoint-Energy JEPA | proposed method |

Looped token and sentence models provide the test-time-compute comparison.

## Mechanism controls

| Control | Removed information or path |
|---|---|
| Counterfactual-only JEPA | endpoint ranking |
| No-counterfactual JEPA | alternative transition supervision |
| Detached endpoint geometry | ranking gradients into predictor |
| True encoded endpoints | rollout drift |
| Shuffled endpoint labels | action-quality correspondence |
| Exact-transition Energy | learned dynamics error |
| Oracle endpoint distance | learned Energy error |
| Symbolic shortest distance | all learned valuation |

## Search controls

- shooting versus genuine beam;
- global versus root-balanced beam;
- terminal endpoint versus cumulative edge score;
- depth and width sweeps;
- fixed requested depth after early solution;
- candidate-order randomization.

## Interface controls

The same trained model should be evaluated with:

1. feasible current and future actions;
2. full current catalogue with feasible future actions;
3. full catalogue at every imagined node.

Do not combine these numbers in one column.

## Fairness

- tune learning rate once per model and dataset;
- report search budgets;
- use identical paired puzzles;
- use approximately matched backbones;
- include a same-backbone objective swap;
- promote only validated recipes to five seeds.
