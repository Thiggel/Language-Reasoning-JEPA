# Token-level causal iGSM JEPA

This subproject removes observed intent actions. A token is the primitive
action; phrase-, sentence-, and longer-span action codes define higher
predictive levels. Its central question is whether multiscale prediction
changes the shared causal state toward useful abstraction and supports
executable hierarchical generation.

Canonical records remain under [`research/hard_text/`](../../research/hard_text/README.md)
to preserve existing links and run provenance. Current work includes fixed
and semantic boundaries, dense multilevel rollout, representation probes,
oracle-terminal planning diagnostics, token support priors, and top-down CEM.

Run families are `hard_hier_*`, `text_hier_*`, `deltajepa_text_*`, and the
controller rounds under `runs/autonomy/`. This project's conclusions must not
be transferred to the observed intent-phrase project without an explicit
experiment.

## Figure

- Discourse / Token-JEPA causal sentence-embedding figure:
  [TikZ source](figures/discourse_token_jepa.tex),
  [rendered PDF](figures/discourse_token_jepa.pdf),
  [SVG](figures/discourse_token_jepa.svg)
