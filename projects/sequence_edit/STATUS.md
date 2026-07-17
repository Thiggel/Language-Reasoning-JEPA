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

The active decision is the minimum useful data and counterfactual coverage,
before returning to hierarchy. The initial one-seed screen crosses 512, 2,000,
and 6,000 unique K=0 trajectories and K={0,1,4,8} at the 2,000-trajectory
anchor, plus one full-shape H4 smoke. Follow-ups already configured but gated
on this screen cover matched update exposure, counterfactual weight, short vs
long repairs, hierarchy, dense rollout, and LDAD removal.
