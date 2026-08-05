# Task and information interfaces

## Interaction

An iGSM prompt describes arithmetic dependencies and asks for one quantity.
The agent chooses an intent phrase. The environment executes it, returns the
numerical outcome sentence, and the agent replans.

```text
prompt -> choose intent -> environment outcome -> choose next intent -> ...
```

The model selects operations. It does not generate their numerical execution.

## State and action

```text
h_t = prompt plus all observed outcome sentences
z_t = learned encoding of h_t
a_t = one language intent
u(a_t) = learned action embedding
```

The environment is deterministic. Training problems are generated freshly.

## Candidate interfaces

| Interface | Real decision | Imagined decision | Label |
|---|---|---|---|
| Feasible menu | prerequisite-satisfied actions | symbolic feasible actions | candidate-privileged for D>1 |
| Full catalogue | every problem action | every problem action | no feasible-action menu |
| Oracle score | any candidates | exact continuation quality | upper bound only |

The validated scaling result uses a feasible current menu and symbolic future
menus. The full-catalogue gate removes both menus. Invalid or repeated actions
then produce:

> The proposed action is invalid and the state is unchanged.

Invalid attempts consume one action and appear in the observed history.

## Information seen by the method

| Component | Prompt/history | Action phrase | True hypothetical outcome | Symbolic quality |
|---|---:|---:|---:|---:|
| State encoder | yes | no | no | no |
| Action encoder | no | yes | no | no |
| Online predictor | through state | through action code | no | no |
| Endpoint head | through latent inputs | through imagined endpoint | no | no |
| Training target builder | yes | yes | yes | no |
| Feasible-tree evaluator | yes | yes | no | no exact quality |
| Full-catalogue evaluator | yes | yes | no | no |

The training target builder may execute sampled counterfactual action
sequences. That is supervision available during training, not test-time access
to hypothetical outcomes.
