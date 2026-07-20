# Cycle: length generalization, action prior, and hierarchy planner pilots

## Decision

Before the final width/depth/learning-rate campaign, determine whether a
candidate-scoring behavioral action prior or the corrected distinct-state
hierarchy belongs in the recipe.  Evaluate both under exact necessary-action
length extrapolation and budgeted success.

## Protocol correction

Unbudgeted success is trivial in stylized iGSM because any policy that keeps
choosing feasible unresolved variables eventually resolves the query.  The
primary curve is therefore success with excess-action budgets 0, 1, 2, and 4.
Training retains the established 3--9 necessary-step distribution.  Test
problems are rejection-sampled at exact lengths 3, 6, 9, 12, 15, and 18 with
strict validation that the requested length was actually obtained.  The
longer cells necessarily use larger graphs and are labelled joint
length/graph-size extrapolation.

## Action-prior pilot

The prior scores every currently feasible intent phrase conditioned on the
current encoded state and normalizes over that menu.  Its inputs are detached,
and its module is initialized after the complete JEPA so enabling it preserves
the same-seed base initialization.  The demonstrated action is the positive;
this is explicit policy supervision.  Compare prior-only, JEPA-only, and
top-2/top-4 prior proposals reranked by JEPA.  Depth 2 and 4 use reference-graph
future feasible menus and are labelled oracle-future-action diagnostics.

Promotion requires top-M to improve budgeted success over both component
controls on at least one out-of-distribution length, without losing strict
in-distribution success.  Prior-only strength changes the interpretation but
does not by itself validate JEPA reranking.

## Hierarchy pilot

`d_macro` is the macro-action bottleneck, not the high-state width.  The high
state remains width 256.  Compare bottlenecks 4, 8, and 16 using separate
low/high encoders, dense four-jump supervision, the learned conditional macro
prior, and 1,200-candidate/20-update CEM.  Direct-code CEM is the support
control; prior-noise CEM is the on-manifold planner.  Each hierarchy is also
evaluated by its own flat low-level planner.

Promotion requires prior-noise CEM to beat the same checkpoint's flat planner
and direct-code CEM, with healthy low/high rank and a consistent signal on
longer problems.  Otherwise hierarchy remains excluded from the flat recipe.

## Validity gates

- exact requested test length or fail loudly;
- full unit suite plus CPU training/planning smokes;
- no dropout and EMA teachers retain existing evaluation-mode behavior;
- deployable and oracle-future-action results remain separate;
- all failed, timed-out, and collapsed cells remain visible;
- no width/depth/LR scale campaign until these architecture pilots resolve.

