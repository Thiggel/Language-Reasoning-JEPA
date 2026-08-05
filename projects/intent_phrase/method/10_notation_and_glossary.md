# Notation

| Symbol | Meaning |
|---|---|
| `h_t` | observed prompt-and-outcome history |
| `z_t` | online latent state |
| `z_0` | initial prompt state |
| `u(a)` | intent embedding |
| `F` | action-conditioned predictor |
| `F^H` | H recursive predictor applications |
| `z_hat_H` | imagined endpoint |
| `g` | EMA solved-state target used only in training |
| `d(z,g)` | normalized mean absolute latent distance |
| `E_H` | lower-is-better endpoint Energy |
| `H` | training horizon |
| `R` | training rollouts per root |
| `D` | test search depth |
| `B` | beam width per first-action root |
| `L_star` | shortest valid solution length |

## Terms

- **Endpoint Energy:** learned scalar for a recursively imagined endpoint.
- **Candidate-privileged:** reference information supplies future candidates.
- **Counterfactual:** alternative action or action sequence from one history.
- **EMA:** slowly updated target network with no gradients.
- **Full catalogue:** every action phrase, including invalid actions.
- **Invalid action:** leaves the environment state unchanged and consumes time.
- **JEPA:** predicts latent targets instead of reconstructing text.
- **Receding horizon:** execute one selected action, observe, and replan.
- **Root-balanced beam:** retains a separate continuation budget per first action.
- **GAR:** historical local action-ranking method; not the primary method.
- **Slack:** allowed excess attempts beyond the shortest solution.

All Energies and distances are lower-is-better.
