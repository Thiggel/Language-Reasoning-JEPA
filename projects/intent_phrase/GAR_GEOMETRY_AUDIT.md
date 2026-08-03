# GAR geometry and generative-bottleneck audit

## Questions

This audit answers two claims that the existing success metrics cannot:

1. Why do the generative token/sentence policies slightly outperform the
   current JEPA planner under the same symbolic feasible-action menu?
2. Does GAR improve JEPA geometry itself, and if so, through which gradient
   path rather than merely through a better value head?

All results remain in the intent-phrase stylized-iGSM subproject. No token-JEPA
or sequence-edit checkpoint or claim is imported.

## Information boundary

Every model receives the same prompt, observed history, and symbolic feasible
intent phrases. Model scores never receive query ancestry, remaining-step
counts, candidate quality, or rendered counterfactual outcomes.

The audit uses privileged information only after scoring:

- query ancestry labels whether a feasible action is necessary;
- true candidate outcomes create an **oracle-transition diagnostic**;
- the solved reference trajectory creates a **terminal-goal diagnostic**.

These controls are not deployable planners and are labeled as such in every
JSON result. The deployed JEPA path remains predicted next state followed by
the learned value function.

## Aligned decision protocol

Models do not follow their own trajectories during the primary comparison.
They score identical states along a canonical necessary-action trajectory.
Where a distractor is feasible, the audit also injects the same deterministic
distractor and scores the immediately following recovery state. This separates
on-expert-history ranking from error recovery without allowing one model's
earlier mistake to change another model's inputs.

For each state the audit reports top-1 necessary-action accuracy, reciprocal
rank, all necessary-versus-distractor pairwise accuracy, signed best-action
margin, and near-tie rates. Results are stratified by trace type, reasoning
step, remaining necessary steps, and selected operation. Pairwise error overlap
shows whether JEPA and LMs fail on the same states.

The sentence latent-MSE LM is scored both ways:

- its paper/deployment score: decoder token cross-entropy;
- its latent next-phrase distance.

If the decoder is good while latent selection is not, its advantage is
specifically generative readout rather than sentence-level state encoding.

## JEPA bottleneck localization

For every feasible action, five lower-is-better scores are computed in the
model's own coordinate system:

1. `gar_teacher_geometry`: EMA-encoded true next state to EMA terminal goal;
2. `online_oracle_geometry`: online-encoded true next state to online goal;
3. `predicted_geometry`: imagined next state to EMA goal, without value head;
4. `oracle_transition_value`: learned value on online true next state;
5. `predicted_transition_value`: deployed value on imagined next state.

The comparisons localize failure:

- poor (1) means the target geometry itself does not order useful actions;
- a drop from (1) to (2) implicates online/EMA mismatch;
- a drop from (2) to (3), plus transition L1/retrieval degradation, implicates
  world-model drift;
- a drop from (2) to (4) implicates the value readout on real states;
- a drop from (4) to (5) implicates predictor/value interaction;
- if (5) exceeds (4), the predictor and value head are usefully co-adapted and
  an oracle-state substitution is not a valid upper bound for that head.

The last case already appears in the five-episode smoke test, which is why the
audit does not assume that replacing imagined states by real states must help.

Additional diagnostics include transition LN-L1, cosine alignment, within-state
next-state retrieval, necessary/distractor goal progress, action-displacement
alignment with the goal direction, effective rank, and aligned linear CKA.
A shared frozen logistic readout uses `[state, goal, state-goal,
abs(state-goal), state*goal]`, the same regularization grid, and separate
train/validation problems for every JEPA variant. It tests information access;
it is not part of model evaluation.

## Why GAR can regularize a geometry that supplies its labels

GAR's labels are stop-gradient distances between true next states and the
terminal goal in the EMA target encoder. Its predicted energies are
`V(F(s,a), s0)`. With `model.value_detach=false`, ranking and advantage-MSE
gradients flow through the value function into `F`, the action encoder, and the
online state representation. The EMA encoder then slowly tracks that online
representation. This is a bootstrapped teacher/student loop, not a fixed
geometric label: the slow target supplies ordering while the online model is
pressured to make imagined transitions and value contours respect it.

That explanation is plausible but the old experiments do not prove it. The
old “no GAR” cell simultaneously removed GAR and counterfactual next-state
prediction. The new factorial therefore contains:

| Cell | Counterfactual latent loss | GAR rank | GAR advantage MSE | GAR/value gradient into body |
|---|---:|---:|---:|---:|
| base | 0 | 0 | 0 | no GAR |
| CF-only | 1 | 0 | 0 | no GAR |
| GAR-only | 0 | 1 | 0.25 | yes |
| full | 1 | 1 | 0.25 | yes |
| detached full | 1 | 1 | 0.25 | **no** |

The detached condition uses the existing `value_detach` architectural switch.
The value head still receives GAR loss and can learn the ordering, but GAR
cannot reshape encoder or predictor states. Counterfactual latent prediction
remains enabled, so full versus detached isolates the GAR body-gradient path
while holding that supervision fixed. The detached cell is run at the anchor
learning rate and two neighboring learning rates because changing gradient
flow can change the optimization optimum.

Evidence for genuine GAR geometry regularization requires all of the following:

1. GAR-only improves head-free predicted geometry over base;
2. CF-only does not explain the entire full-model gain;
3. full improves over detached full in head-free geometry or transition
   alignment, despite both receiving identical GAR labels and CF supervision;
4. the conclusion survives the detached LR cross-check;
5. gains appear on forced-error states, not only expert histories.

If only deployed value ranking improves, the defensible claim is value/dynamics
co-adaptation, not general geometric regularization. If true-next geometry
improves but predicted geometry does not, GAR reorganizes representations but
does not fix world-model drift. If the frozen shared readout improves while the
native value does not, the information is present but the scorer is the
bottleneck.

## Reproducible entry points

- `scripts/audit_intent_gar_geometry.py`: JEPA five-stage localization and
  feature export.
- `scripts/audit_intent_lm_decisions.py`: token/sentence native decision traces.
- `scripts/compare_intent_gar_geometry.py`: frozen readouts, CKA, and matched
  mechanism comparison.
- `scripts/compare_intent_policy_decisions.py`: aligned error overlap and
  difficulty strata.
- `scripts/run_intent_decision_audit_cell.sh`: cluster-safe wrapper.
- `src/textjepa/analysis/intent_decisions.py`: tested ranking and transition
  metrics.

The submitted round is
`2026-08-03-intent-decision-and-gar-mechanism-audit-v1`.
