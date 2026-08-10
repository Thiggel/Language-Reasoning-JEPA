# Charter

## Scientific sequence

The project tests one implication at a time:

`predictive sufficiency -> stable recurrent state -> goal-directed geometry`

- H1: higher-layer state plus the realized next token predicts the next
  contextual lower-layer state better than action-only and no-action controls.
- H2: joint training makes the predictor-removed full model more
  transition-sufficient without collapse and improves NLL or downstream
  performance beyond token- and compute-matched NTP-only LoRA.
- H3: predicted layer-8 state can be recursively advanced through layers 9–16
  with bounded open-loop error and useful speed or KV-memory savings.
- H4: an explicitly trained temporal geometry adds progress/value information
  beyond position, likelihood, entropy, and a direct value model.

## Validity gates

Stage 1 admits a positive representation claim only if held-out fresh-probe
transition error improves, the one-state/history-window gap falls by at least
20%, effective rank does not collapse, and the predictor-removed LM improves
NLL or a downstream aggregate. Auxiliary training loss alone is not evidence.

Stage 2 admits a recurrent-decoding claim only if error remains controlled to
at least 128 actions, task retention is at least 90%, excess NLL is below 0.3
nats/token, and measured throughput or KV-memory improves materially. Periodic
refresh and speculative verification are separate operating points.

Stage 3 admits a planning claim only after the scorer improves matched-depth
branch ranking and exact-state planning. Predicted-state planning is evaluated
only with the same frozen scorer. Oracle terminal-state distance is never
reported as an inference algorithm.

## Negative controls

The minimum Stage 1 comparison contains NTP-only LoRA, no-action, action-only,
same-layer dynamics, NITP-style projection, frozen-backbone prediction, and an
equal-FLOP NTP control. Stabilizers receive fair coefficient ranges and an LR
cross-check whenever gradient scale changes materially.
