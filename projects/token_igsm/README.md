# Token-level causal iGSM JEPA

This subproject removes observed intent actions. It pivoted on 2026-07-30 to
the staged [hierarchical predictive-state planning specification](REVISED_PLAN.md).
A frozen causal LM supplies wide hidden states and executable proposals. The
controlled states form the strict nested tower
`h -> E0 -> z0 -> E0_to_1 -> z1`; the sentence state is a further abstraction
of the token state, not an independent projection of the LM hidden state.
The first decision is whether token JEPA supports oracle-goal planning with an
exact-endpoint control.

The [normative implementation contract](NORMATIVE_CONTRACT.md) resolves
encoder notation, prefix indexing, causal pre-action context, the three
oracle goals, stage ordering, pinned Qwen constants, counterfactual policy,
and iGSM splits. It
supersedes conflicting ambiguous passages in the earlier plan.
The exact-capacity and exact-trainable-parameter Qwen controls are fixed in
[`PARAMETER_MATCHED_QWEN_BASELINES.md`](PARAMETER_MATCHED_QWEN_BASELINES.md).

Canonical records remain under [`research/hard_text/`](../../research/hard_text/README.md)
to preserve existing links and run provenance. Those records are legacy. New
work is validity-gated: token oracle planning, then a true sentence-waypoint
worker, then and only then macro-actions, high-level CEM, and unbudgeted value
distillation.

Official staged interfaces:

- `collect_hierarchical_language_features.py` for pinned, sharded
  Qwen3.5-0.8B features and flattened counterfactual batches; its default
  shared-cache engine prefills each root once and dynamically prunes finished
  branches, while retaining independent full exact re-encoding;
- `train_hierarchical_language_jepa.py --init-checkpoint ... --admission ...`
  for cumulative training stages;
- `evaluate_hierarchical_language_oracles.py` for evaluation-only gates;
- `write_hierarchical_admission.py` to bind measured thresholds to an exact
  checkpoint and dataset fingerprint;
- `build_hierarchical_value_teacher.py` for total-prefix oracle value replay;
- [First full strict-nested language-planning experiment](FULL_HIERARCHICAL_LANGUAGE_EXPERIMENT.md)
  for the manager-to-worker MPC contract, oracle/no-terminal evaluations, and
  isolated Euclidean/Mahalanobis × raw/quasimetric ablation;
- `diagnose_hierarchical_language.py` for representations, probes, shortcuts,
  and counterfactual diagnostics.
- `generate_hierarchical_igsm_pool.py` and
  `build_hierarchical_igsm_splits.py` for constructive verified ID,
  length-OOD, structural-OOD, and paraphrase-OOD manifests;
- `build_flat_token_oracle_candidates.py` for frozen-LM proposals, learned
  `P0` rollout, and exact Qwen re-encoding in one bound artifact;
- `run_hierarchical_language_pilot.py` for the bounded first-five-mechanism
  pilot, and `plot_hierarchical_planning_effort.py` for accuracy-versus-effort
  plots.

Controlled sentence-endpoint geometry uses matched triplets: the same prefix
followed by an observed reasoning step, a meaning-preserving paraphrase, or a
minimally changed incorrect result. Standalone logical triplets (for example,
“all swans are white” versus “no swan is white”) are reported separately as
qualitative out-of-domain controls. Stage-1 plots are explicitly labeled
`z0_sentence_boundary`; `z1_sentence` is not exported until the nested sentence
encoder has actually been trained. Quantitative triplet distances are
authoritative, while t-SNE/UMAP coordinates are qualitative views.

Run families are `hard_hier_*`, `text_hier_*`, `deltajepa_text_*`, and the
controller rounds under `runs/autonomy/`. This project's conclusions must not
be transferred to the observed intent-phrase project without an explicit
experiment.

Current bounded evidence and the readable planning-effort plot are linked from
[`STATUS.md`](STATUS.md). The run is explicitly candidate-privileged and
single-seed until the queued Alex replication completes.

## Figure

- Discourse / Token-JEPA causal sentence-embedding figure:
  [TikZ source](figures/discourse_token_jepa.tex),
  [rendered PDF](figures/discourse_token_jepa.pdf),
  [SVG](figures/discourse_token_jepa.svg)
