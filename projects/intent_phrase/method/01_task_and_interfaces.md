# Task and information interfaces

## The task in ordinary language

An iGSM problem describes a small dependency graph in natural language. Some
quantities are given directly. Other quantities are defined by arithmetic
operations over earlier quantities. One quantity is queried.

At each reasoning step, the agent does not write the numerical conclusion. It
chooses an intent phrase such as:

> derive the blue boxes from the red boxes plus the green boxes

The environment executes that intent, computes the numerical result, and
returns an outcome sentence. The agent then chooses the next intent. Success
means reaching the queried result within the allowed number of actions.

This separates two abilities:

- choosing which operation should be performed;
- generating or executing the operation's textual and numerical outcome.

The intent-phrase project studies the first ability. The environment supplies
the second.

## Underlying objects

We use the following plain-text notation:

- `p`: one generated problem.
- `x_0`: the natural-language prompt containing definitions and the question.
- `a_t`: the intent phrase selected at step `t`.
- `y_(t+1)`: the outcome sentence returned after executing `a_t`.
- `h_t`: the complete observed history before choosing `a_t`.
- `g`: the task goal, namely resolving the queried variable.

The history is:

```text
h_t = (x_0, a_0, y_1, a_1, y_2, ..., a_(t-1), y_t)
```

The model encodes this history into a latent state `z_t`.

## Semantic actions versus language actions

The environment has semantic operations such as resolving a particular graph
node. The model sees a language realization of that operation. Different
phrases could express the same semantic intervention, although the current
stylized generator uses a mostly canonical action template.

This is why the paper describes language as a non-canonical interface rather
than as the world state itself. A useful representation should preserve the
consequences of an action while becoming less sensitive to irrelevant wording.

## What is feasible?

An action is symbolically feasible when all parent quantities needed by its
operation have already been resolved. The currently feasible menu can be
computed exactly from the generated dependency graph.

There are several distinct action interfaces in the repository:

### Current feasible menu

At each real environment state, enumerate only actions whose prerequisites are
satisfied. Every compared model scores this same menu. This is the main
controlled planning interface.

**Privilege label:** symbolic feasible-action menu.

### Future feasible menus

During lookahead deeper than one, the evaluator can use the symbolic graph to
enumerate which actions would become feasible under a hypothetical action
sequence.

**Privilege label:** candidate-privileged or oracle future-action tree. The
scores can still come entirely from the JEPA, but candidate availability uses
the reference graph.

### Full action catalogue

Enumerate every intent in the problem, including currently infeasible ones.
This tests proposal and feasibility learning as well as consequence ranking.
It is a different scientific question and should not be mixed with the shared
feasible-menu result.

### Oracle action quality

Use exact remaining-step distance, exact relevance, or the ground-truth
shortest path to score actions.

**Privilege label:** oracle score. This is a diagnostic upper bound, never the
learned method.

## What each component can see

| Component | Prompt | Observed outcomes | Candidate phrase | Symbolic feasible menu | Exact action quality |
|---|---:|---:|---:|---:|---:|
| Online encoder | yes | yes | no | no | no |
| Action encoder | no | no | yes | no | no |
| JEPA predictor | through `z_t` | through `z_t` | through action code | no | no |
| Energy head | through latent inputs | through latent inputs | only through predicted consequence | no | no |
| H>1 training teacher | through EMA states | observed prefix only | yes | yes, for continuation expansion | no |
| D>1 evaluator | through online states | observed prefix and imagination | yes | yes, for future expansion | no |
| Exact-distance control | yes | yes | yes | yes | yes |

## Determinism and partial observability

The current stylized iGSM environment is deterministic. Given the problem,
resolved variables, and an action, the next symbolic state and outcome are
fixed. The broader paper language allows history-based partial observability,
but the present experiments do not establish performance in a stochastic or
multimodal environment.

## Dataset generation

Training problems are generated procedurally. A nominal training epoch has
30,000 examples in the current Energy experiments. The fresh-epoch sampler
uses a disjoint index range on each epoch, giving 300,000 fresh generated
examples over ten epochs rather than repeating a fixed 30,000-example file.

The current default problem range contains approximately 3–9 necessary
reasoning steps, 6–12 variables, and up to two distractors. Therefore a beam
depth of 16 usually saturates before sixteen meaningful actions. It is a
test-time-compute and stability control, not proof of sixteen-step length
generalization.
