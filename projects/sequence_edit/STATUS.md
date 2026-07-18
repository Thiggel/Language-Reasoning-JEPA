# Sequence-edit status

Current result (2026-07-18): the token-aligned structured primitive passes
operation, current-pointer, and local-content causal gates. Fixed mixed EMA is
the default; 18k presentations replicate across three seeds, d512 improves
matched width controls, and exact counterfactual breadth saturates at K=1.
K=1 slightly harms recursive rollout, so K=0 remains the rollout default.

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

That restart is now implemented. Primitive actions use an operation plus a
pointer to a current token/gap and an optional content token; transitions
retain token latents and recursively apply the same zero-dropout spatial
predictor. Mask, random-replacement, removal, and temporal-curriculum data
paths exactly recover the terminal buffer. EMA targets are forced to remain
in evaluation mode. A CPU train/evaluate/audit fixture and 19 focused tests
pass. The active GPU decision is the four-corruption by three-stabilizer pilot
plus H=1/H=4 goal-advantage distillation; hierarchy remains gated on primitive
action sensitivity.

The curriculum VICReg and faithful text LDAD screens have now completed. LDAD
substantially improves one-step and recursive prediction and preserves rank,
but every tested coefficient remains below the shuffled-action causal gate.
The earlier mask-only action signal is also distribution-specific: a small
cross-evaluation loses it on mixed edits. A data audit found that
non-curriculum runs repeated corruptions despite `fresh_per_epoch: true`; the
dataset path and cross-corruption audit are now fixed and tested. The next
decision is a matched fixed-versus-fresh mixed/mask/curriculum comparison plus
one lower-learning-rate LDAD-20 control, not hierarchy or planning.
