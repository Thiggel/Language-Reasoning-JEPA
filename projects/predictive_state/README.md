# Action-conditioned predictive-state language models

This subproject tests whether a causal LM can consolidate prefix information
into a state that is both easy to update after a known token action and useful
on the ordinary full transformer path.

The initial transition is:

`(h_t^12, h_t^16, x_(t+1)) -> predicted h_(t+1)^8`

for OLMo 2 1B, with the layer-fraction-matched
`(h_t^18, h_t^24, x_(t+1)) -> predicted h_(t+1)^12` screen on Qwen2.5-0.5B.
The claims are deliberately staged:

1. representation shaping on the full LM path with the predictor removed;
2. recurrent execution through only the predictor and upper transformer;
3. oracle progress diagnostics, then explicitly learned goal distance;
4. exact-state planning before any predicted-state planning claim.

Predictive-state supervision does not identify a unique metric. Raw terminal
distance is therefore an oracle diagnostic, not a value function. Stage 3
uses an explicit temporal metric and a separately trained direct-value
baseline before planning or reward-shaping conclusions are admitted.

Entry points:

- [`DESIGN.md`](DESIGN.md): complete scientific and implementation protocol.
- [`CHARTER.md`](CHARTER.md): falsifiable questions and validity gates.
- [`STATUS.md`](STATUS.md): current observed state only.
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md): immutable run-family index.
- [`ARTIFACTS.md`](ARTIFACTS.md): code/config/run ownership.
- [`DECISIONS.md`](DECISIONS.md): durable design decisions.
- [`AUDIT.md`](AUDIT.md): adversarial implementation passes and regressions.
- [`QUESTION_BACKLOG.md`](QUESTION_BACKLOG.md): unresolved questions.

The implementation is shared under `src/textjepa/`; run artifacts belong under
`runs/autonomy/predictive_state/<round-id>/<job-id>/`. Stage 3 trajectory data
or reports must state whether terminal states, verified outcomes, candidate
sets, or information from another subproject are oracle, symbolic,
candidate-privileged, or cross-project inputs.
