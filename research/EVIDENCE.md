# Cumulative evidence ledger

This file contains only decision-grade summaries. Follow links for details;
do not copy raw logs here.

| ID | Observation | Status | Scope | Source |
|---|---|---|---|---|
| E001 | Corrected root-balanced discrete hierarchical planning did not beat the flat one-step controller in the frozen 3×500 confirmation. | supported | observed-intent iGSM | [`HANDOFF.md`](../HANDOFF.md) |
| E002 | Earlier apparent hierarchy gains depended on lexicographic proposal truncation and were invalid. | supported | discrete hierarchy evaluator | [`research/intent_phrase/waves/11_controller_outcomes_and_discrete_hierarchy.md`](intent_phrase/waves/11_controller_outcomes_and_discrete_hierarchy.md) |
| E003 | The faithful iGSM reference action order can make an always-first policy solve the benchmark; stable problem-specific shuffling is required. | supported | faithful iGSM evaluation | [`HANDOFF.md`](../HANDOFF.md) |
| E004 | The non-symbolic reduced observed-intent JEPA is competitive but below its matched token intent-policy control; this does not establish discovered text hierarchy. | supported | observed-intent iGSM | [`README.md`](../README.md) |
| E005 | Cross-cluster experiment execution currently lacks one uniform compact validity/result contract. | observed engineering gap | all tracks | [`cycles/hierarchical-language-jepa/2026-07-16-autonomy-bring-up.md`](cycles/hierarchical-language-jepa/2026-07-16-autonomy-bring-up.md) |
| E006 | The hard-text token hierarchy's configured VICReg term was evaluated on a no-grad EMA target and therefore exerted zero encoder gradient; historical runs did not test active VICReg. | supported implementation audit | hard-text hierarchy | [`research/cycles/hard_text/2026-07-16-hierarchy-gradient-abstraction.md`](cycles/hard_text/2026-07-16-hierarchy-gradient-abstraction.md) |
| E007 | In the common-protocol causal intent-phrase matrix, two-step latent-goal preference distillation produces the dominant gain (.125 to .588 strict), but the resulting JEPA remains below the matched token intent policy (.827 strict). | supported validation result | observed intent-phrase iGSM | [`projects/intent_phrase/STATUS.md`](../projects/intent_phrase/STATUS.md) |
| E008 | Corrected observed-intent hierarchy confirmations are negative and hierarchy is excluded from the intent-phrase paper recipe. | supported | observed intent-phrase iGSM | [`research/intent_phrase/waves/11_controller_outcomes_and_discrete_hierarchy.md`](intent_phrase/waves/11_controller_outcomes_and_discrete_hierarchy.md) |
| E009 | All five faithful token-edit v3 jobs failed before optimization and therefore provide no hierarchy evidence. The failure exposed that splitting official tokenized solutions on standalone `.` collapsed multi-step solutions into one chunk. | invalid run plus supported implementation audit | sequence edit | [`cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md`](cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md) |
| E010 | After preserving official step boundaries, a 128-example data audit recovered every clean terminal buffer, collapsed no multi-step solution, and found synthetic repair paths near minimum token-edit length (mean ratio 1.029). | supported data audit; no model claim | sequence edit | [`cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md`](cycles/sequence_edit/2026-07-16-faithful-token-hierarchy.md) |
| E011 | The process-valid six-cell faithful edit pilot is action-blind: shuffled-action error ratios are 1.00000–1.00014, most cells lose to persistence, and K=4 lowers effective rank by 24.5% at the matched 2k anchor. Global same-state targets are nearly invariant to a one-token edit (.000228 pairwise LN-L1), while changed-step targets are well separated (.634). | supported one-seed validity result; local-target remedy untested | sequence edit oracle denoising | [`2026-07-17 report`](reports/sequence_edit/2026-07-17-faithful-edit-data-counterfactual-pilot/REPORT.md) |

Use these confidence labels: `supported`, `provisional`, `contradicted`,
`invalid`, and `unknown`. A completed process is not automatically supported
scientific evidence.
