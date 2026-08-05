# Notation and glossary

## Core notation

| Symbol | Meaning |
|---|---|
| `p` | Generated reasoning problem |
| `x_0` | Natural-language prompt and question |
| `a_t` | Intent phrase chosen at step `t` |
| `y_(t+1)` | Outcome sentence after executing `a_t` |
| `h_t` | Full observed history before action `a_t` |
| `z_t` | Online latent state encoding of `h_t` |
| `z_t^ema` | EMA target encoding of the same history |
| `u(a)` | Compact action code for intent `a` |
| `F` | Action-conditioned JEPA predictor |
| `z_hat_(t+1,a)` | Predicted successor after candidate action `a` |
| `g` | EMA-encoded terminal goal state used by the teacher |
| `d(z,g)` | Normalized mean absolute latent distance |
| `E` | Lower-is-better transition Energy head |
| `V` | Lower-is-better state Energy head |
| `H` | Training teacher horizon |
| `D` | Test-time planner beam depth |
| `B_teacher` | Beam width maintained separately per training root |
| `B_eval` | Global evaluation beam width |
| `L_star` | Minimum necessary action count |
| `excess` | Model steps minus `L_star` |

## Terms

- **Action code:** Compact learned vector representing an intent phrase.
- **Action catalogue:** All intent actions defined by a problem, including
  potentially infeasible ones.
- **Advantage target:** Geometric endpoint distance minus current distance.
  It is a progress surrogate, not automatically a Bellman advantage.
- **Candidate-privileged:** Uses reference information to enumerate future
  candidate actions, while learned components may still score consequences.
- **Causal predictor:** Predictor that retains ordered state-action history and
  cannot attend to future positions.
- **Counterfactual:** An alternative action and its hypothetical consequence at
  the same current state.
- **Direct ranker:** Scores a state-action pair without predicting a successor.
- **Discourse state:** Latent summary of the prompt and outcomes observed so far.
- **EMA:** Exponential moving average target network.
- **Energy:** Lower-is-better learned scalar used to compare candidates.
- **Feasible action:** Action whose symbolic prerequisites are resolved.
- **GAR:** Geometric Advantage Ranking, the pairwise action-ordering loss.
- **Geometry:** Distances, directions, neighborhoods, and order relations among
  latent states.
- **JEPA:** Joint-Embedding Predictive Architecture; predicts target
  representations rather than reconstructing raw tokens.
- **Latent:** Internal vector representation.
- **Oracle:** Uses exact hidden environment quality or outcome information not
  available to the learned deployment method.
- **Receding horizon:** Imagine several actions, execute one, observe, replan.
- **Root-balanced beam:** Separate continuation budget under every first action.
- **Slack:** Number of excess actions permitted beyond the shortest solution.
- **State Energy:** Scalar quality of an endpoint state.
- **Symbolic menu:** Feasible-action set computed from the reference graph.
- **Teacher horizon:** Number of actions considered when producing the target
  attached to a candidate first action.
- **Terminal composition:** Score a beam only by the Energy at its final edge or
  endpoint.
- **Transition Energy:** Scalar quality assigned using a predecessor and
  successor state.
- **True-state ablation:** Energy head is trained on EMA true successors rather
  than online predicted successors.
- **VICReg:** Variance-invariance-covariance regularization used to resist
  representation collapse.

## Sign convention

Every current Energy and distance is lower-is-better.

```text
smaller distance  = closer to goal
smaller Energy    = preferred action or beam
negative advantage target = progress toward goal
```

Any document or plot using higher-is-better scores must explicitly state the
sign conversion.
