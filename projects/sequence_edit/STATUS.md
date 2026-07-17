# Sequence-edit status

The original faithful hierarchy round is scientifically invalid: all five
jobs failed before optimization, and the apparent 276-token "sentence" exposed
a data-boundary bug. Official iGSM steps usually end in fused punctuation, so
the old adapter collapsed whole multi-step solutions into one chunk.

The adapter now preserves official nested steps and exactly recovers the clean
terminal buffer after literal token edits. The actual task is explicitly
labelled synthetic oracle denoising: gold-solution tokens define corruptions
and the inverse repair path. Counterfactual data now contains only sampled
current-buffer edits and their mechanically exact outcomes, without preference
or target-relative quality labels.

The six-cell data/counterfactual screen completed, but every cell is
action-blind: shuffled actions change error by less than 0.015%, and most
predictors are worse than copying the current buffer. K=4 lowers raw internal
error but loses 24.5% effective rank and still fails both causal controls, so
no K or data anchor is selected.

The mechanism audit found that exact global next-buffer targets for four edits
from the same state are separated by only 0.000228 normalized L1. Exact
changed-step targets are separated by 0.634. The active decision is therefore
whether local mechanically exact outcome targets restore action sensitivity.
Coefficients 0.25, 1, and 4 plus a high-weight learning-rate cross-check all
completed, but assignment remains at chance, shuffled actions change error by
at most 0.15%, every predictor loses to persistence, and rank drops 11.6--26.4%.
The pooled state is therefore rejected under this recipe. The final bounded
interface test exposed current official-step embeddings directly to the
action-conditioned attention predictor. Assignment improved only to 27.56%,
shuffled actions slightly improved error, prediction still lost to persistence,
and rank fell 23.4%. This state family is retired. K=8, matched exposure, more
data, hierarchy, dense rollout, and LDAD removal will not run. Any future
restart requires a new token-aligned recursive interface and CPU causal fixture.
