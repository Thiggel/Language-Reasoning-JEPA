# Wave 07 - representation probes and interpretability

## Completed probes

- current and lagged value decoding;
- final answer emergence over trace position;
- resolved-set membership;
- ancestor/query relevance;
- operation and newly resolved variable;
- modular circular-code probes;
- action and counterfactual transition matching;
- effective rank, variance, and trajectory geometry.

## Findings to date

| Feature | Finding |
|---|---|
| Working memory | current value ~0.90; decays approximately 0.43/0.29/0.21 by lag |
| Structure | resolved membership ~0.82; query ancestry ~0.76 |
| Answer | low mid-trace, becomes fully decodable when query resolves |
| Modular value | partially circular representation, ridge R2 ~0.52 in grounded model |
| Query identity | weakly explicit despite strong relational relevance |
| Planning relevance | value/action ordering predicts end-to-end chain success |

## Backlog after hierarchy selection

- layerwise and hierarchy-level probes;
- linear versus nonlinear decodability of syntax, entities, operations,
  values, dependency depth, remaining subgoals, distractor status, and
  uncertainty;
- minimum-description/MDL controls and selectivity against random labels;
- causal interventions along probe directions;
- compare information retained in token, phrase, sentence, and macro states;
- test whether features that improve `V_hi` and subgoal reachability are the
  same features decoded by probes.

The representation analysis is a primary paper contribution, but final probe
runs wait for the selected hierarchy so they are not repeated on obsolete
checkpoints.
