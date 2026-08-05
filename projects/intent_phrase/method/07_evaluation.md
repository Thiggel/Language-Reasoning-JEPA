# Evaluation

## Primary metric

An episode succeeds when the queried quantity is resolved.

```text
excess actions = executed attempts - shortest valid solution length
strict success = excess actions = 0
slack-k success = excess actions <= k
```

The final evaluator should run once, record first solution time, and derive the
complete excess-action curve. Current artifacts separately report slack 0 and
2.

## Paper protocol

| Item | Requirement |
|---|---|
| Training seeds | 5 |
| Test puzzles | at least 1,000 paired episodes |
| Depths | `1,2,4,8,16` |
| Beam widths | `1,2,4,8,16` |
| Uncertainty | paired bootstrap intervals |
| Selection | validation only |
| Final reporting | held-out test only |

## Interface labels

Every result must state one of:

- current feasible menu only;
- symbolic future feasible tree;
- full action catalogue;
- oracle transition or score.

## Metrics

| Metric | Meaning |
|---|---|
| Strict success | shortest-path solutions |
| Slack curve | tolerance to extra attempts |
| Invalid-action rate | feasibility without a menu |
| Distractor rate | valid but unnecessary actions |
| Pairwise endpoint rank | local Energy correctness |
| Top-1 root recall | best first action retained |
| Rollout error by depth | imagination drift |
| FLOPs | actual test-time compute |

## Generalization

Train on bounded reasoning length. Test on longer necessary lengths and more
distractors. Report curves rather than one pooled OOD number.
