# Cycle: deterministic targets and support-constrained search

## Decision

After removing stochastic EMA targets, can calibrated geometric advantages and
a hard token-support constraint turn the hierarchy's predictive signal into
executable planning?

## Validity corrections

- The base token encoder accidentally inherited Transformer dropout 0.1, and
  the registered EMA teacher entered training mode with its parent. New cells
  use zero encoder dropout and a permanently evaluation-mode EMA target.
- Prior shooting previously compared primitive states directly with a distinct
  higher-level subgoal. New planning causally lifts primitive rollouts into the
  target level before scoring them.
- GAR now predicts the change in EMA latent goal distance and combines
  pairwise ranking with optional MSE calibration.

## Minimal experiment

Four matched no-GAR cells screen dense rollout depth and weighting after the
dropout correction: depth 1, depth 4 uniform, depth 4 with discount 0.5, and
depth 8 with discount 0.7. Two matched GAR cells compare ranking alone against
ranking plus advantage MSE using detached token priors and distinct hierarchy
states.

The calibrated cell evaluates topmost-only value guidance followed by pure
latent subgoal matching. Primitive search is restricted to the prior's top-k
support and compared across categorical CEM, beam, bounded A*, and PUCT.
Macro search compares hard conditional-codebook beam/PUCT with conditional
prior progressive-widening PUCT. All searches see the same oracle terminal
encoding, never symbolic feasibility labels or an auxiliary language model.

## Primary evidence and gates

- Prediction drift by token and macro horizon, target determinism, effective
  rank, and representation probes are health gates.
- GAR pair accuracy, advantage MSE, reference-token rank, and calibration are
  separated from planning results.
- Planning reports reference-token top-1/top-5/top-20 recall of the fitted CEM
  distribution, valid and invalid sentences, goal distance, drift, and solved
  episodes.
- Continue a planner only if it improves executable actions or reference recall
  without worse drift or optimizer exploitation. Scale nothing from this
  single-seed two-episode screen.

## Human steering incorporated

The human requested numerical advantage calibration beside ranking,
topmost-only value use, hard prior/codebook support, prior-top-20 categorical
CEM with iterative refinement, and matched tree-search alternatives. These
requirements define this cycle. A goal-conditioned token prior remains the
next conditional step if hard top-k support has high recall but poor selection.
