# Hard-text hierarchy backlog

Only hierarchy model/planner work may be launched until Stage T3 has a valid
end-to-end result.

## Stage T0 - pause the variational tracks (complete)

- Stop observed-action probabilistic, latent-action, counterfactual-prior,
  and Delta-JEPA-text jobs.
- Preserve all artifacts and document their completed conclusions.
- Use a deterministic EMA+VICReg base for the first hierarchy screen so the
  hierarchy is the only changing mechanism.

## Stage T1 - fixed-span multiscale world model (submitted)

Token `x_t` is the primitive action and the encoded prefix state is `s_t`.
Each higher action is constructed exactly from the ordered lower-level action
sequence in its span: project each lower action, concatenate in temporal
order, and project to a shared macro-action bottleneck.

Initial levels:

| Level | Transition scale | First screen |
|---|---|---|
| 0 | one token | fixed |
| 1 | short phrase | 4, 6, 8, 10 tokens |
| 2 | sentence-like | 20, 24, 30 tokens |
| 3 | paragraph-like | 64, 96, 128 tokens |

First select Level-1 span and bottleneck, then add Level 2, then Level 3.
Do not run the full Cartesian product. At each selected span, test macro
dimension `{8,16,32,64}` and a capacity-matched flat model.

Wave 05 is queued with controlled Level-1 and two-level screens plus automatic
representation probes.

After fixed spans work, compare semantic boundaries: punctuation/phrase,
sentence, and paragraph delimiters. Fixed spans remain the controlled main
ablation.

## Stage T2 - deterministic versus probabilistic macro actions

At the selected spans/dimensions compare:

1. deterministic projected concatenation plus a fitted conditional Gaussian
   density;
2. Gaussian posterior `q(m | lower actions)` plus state-conditioned prior
   `p(m | s)`;
3. optional mixture prior only if a single Gaussian fails same-state coverage.

The distribution does not automatically identify off-manifold states. Use
conditional negative log density and high/low-model disagreement as separate
support diagnostics and planning penalties.

## Stage T3 - hierarchical planning (active)

Test both requested interfaces:

1. **Support-constrained text-span planning:** the model's own
   state-conditioned token prior proposes spans to the next boundary; the
   corresponding macro-action is evaluated by the high-level predictor and
   value; execute the first span and replan. An auxiliary LM is excluded from
   the primary experiment.
2. **Top-down latent-macro planning:** prior shooting or prior-regularized CEM
   produces a high-level subgoal; token beam search finds a lower-level span
   whose predicted state matches that subgoal.

Compare flat token beam, but all proposed JEPA variants must use a high-level
model. Sweep high horizon `{1,2,4}`, proposal count `{8,32}`, and token beam
`{4,8,16}` under matched compute. Report latent-macro CEM only with HWM-scale
budgets: 900--3000 high-level candidates, 15--40 refits, fixed elite counts,
and EMA-smoothed variance. Smaller runs are engineering smokes only. The
implemented top-down CEM defaults to 1,000 candidates, 20 refits, and 100
elites, recursively turning macro predictions into lower-level subgoals.

Because the correct terminal reasoning state is unknown at inference, first
run an oracle-terminal-latent diagnostic, then train a query-conditioned
terminal energy from completed traces and counterfactual/partial states.

## Stage T4 - dense recursive shifted-sequence supervision

For each level, encode the complete state sequence, predict the next shifted
sequence with its causal predictor, feed that predicted sequence back into
the predictor, and repeat. Supervise every valid shifted target:

`L_N = sum_i lambda^(i-1) MSE(S_hat^(i)[i:], sg(S[i:]))`.

Sweep `N={1,2,4,8}` and `lambda={1,.9,.7,.5}` at Level 0 and at every active
higher level. Report direct high-level error against recursively composed
lower-level error.

## Stage T5 - full study

- semantic versus fixed boundaries;
- hierarchy levels and strides;
- macro dimension/distribution;
- planner and planning depth;
- selected regularizer/target factorial;
- layerwise and hierarchy-level representation probes;
- three seeds for the selected model and final ablations;
- matched token/sentence LM baselines and qualitative trajectories.
