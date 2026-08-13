# Token-level iGSM status

_Last update: 2026-08-13._

## Current state (2026-08-13)

The binding constraint is now localized and it is **not** where the last several
cycles looked. Full detail:
`research/reports/token_igsm/2026-08-13-worker-context-coverage/REPORT.md`.

**No checkpoint in this track has ever passed its admission gates** — every
training run ends in `validity_gate_stop`. Four value-continuation variants
(euclidean/mahalanobis x vicreg/sigreg x two LRs) fail identically at ~5%
`worker_executability` (0.047–0.060, gate needs 0.10) with
`oracle_success_gain = 0.0`, while `optimizer_curse_regret` swings 2.2–24.7.
Nothing upstream of the worker moves the number.

The **sentence level is sound** on those same checkpoints: `symbolic_state_purity
1.0`, `effective_rank 23.6`, `exact_symbolic_waypoint_success 1.0` (predicted
0.859), `heldout_sentence_dynamics_gain 0.967` vs identity, commutation 0.104 vs
0.176 baseline. The manager knows where to go; the worker cannot write text that
gets there.

**Why the worker fails (2026-08-13 2x2, frozen-reference proposal coverage, no
checkpoint loaded, no planner):** format is inherited from context, not from
instructions. `oracle@32` by prefix x prompt —

| prefix | pinned prompt | canonical prompt |
| --- | --- | --- |
| canonical (teacher-forced) | 0.727 | 0.742 |
| self_generated (the real planning regime) | **0.000** | 0.039 |

Changing only the prefix collapses the worker 0.727 -> 0.000; 0 of 4096
continuations parse. Instructing the format recovers 5% of the loss. This is a
**format lock-in, not an error-recovery failure**: the model's content is often
correct, but its first free-run line is a markdown header and its own context
keeps it in markdown thereafter. Off-format text is both unscoreable by the
verifier and off-manifold for the JEPA states.

Charter decision 2 ("can a token worker realize a true future sentence
waypoint") is answered **no for a frozen zero-shot Qwen**, with a mechanism.

## Ruled out as the binding constraint

JEPA representation (gates pass); metric / regularizer / learning rate (four
variants identical); planner compute (K1=4 = K1=1; beam spends 269k transition
evaluations for the worst text of any worker); prompting (the 2x2 above).

## Next (priority order)

1. **Prefix priming** — seed the assistant turn with one canonical line rather
   than instructing in the system prompt. ~30 min, frozen LM untouched, charter
   intact. Label the seed line: teacher-forced = privileged; templated from the
   problem text = deployable.
2. **Constrained decoding** onto the canonical grammar — decisive either way.
   Parse -> 1.0 with validity climbing toward 0.73 means the 5% wall was a
   formatting artifact; validity staying near 0 means the content is genuinely
   absent off the teacher-forced path.
3. **Fine-tuning the generator** — likely effective but changes the charter's
   question ("derived from a frozen reasoning LM"). Hold until 1 and 2 decide.

Pair 1 with a parse-gate on worker candidates (the "supported proposals" idea in
`FULL_HIERARCHICAL_LANGUAGE_EXPERIMENT.md`) so the JEPA cost cannot select an
off-manifold step.

## Measurement warnings for whoever runs this next

- **Final-answer accuracy is a termination metric here.** The 2026-08-10
  full-episode pilot returned 0/8 for all five workers because **0 of 40
  episodes emitted a `\boxed{}` at all** within the 8-step budget; at 16 steps
  the earlier gate reached one, and that one episode is the one scored correct.
  Use `step_parse_rate` / `step_validity_rate` (added 2026-08-13) alongside it.
- `greedy` / `oracle@1` in coverage outputs is sample-0 at temperature 0.8, not
  argmax — SE ~0.044 at 128 roots. `oracle@32` is the robust column.
- `run_hierarchical_mpc_search_matrix.py` had `--manager-grounding none`
  hard-coded until 2026-08-13, so every matrix run before that date has
  **unmeasurable** optimizer-curse regret (reported as 0.0; now `null` plus a
  sample count). Do not read those zeros as "no curse".
- `mean_worker_exact_gap` is 0.0 by construction for search workers.

## Superseded history

Fixed-span and semantic-boundary hierarchy experiments showed higher-level
prediction changes representation diagnostics without executable planning
benefit. Oracle-terminal and support-constrained planning results are
candidate-privileged diagnostics, not deployable claims. Earlier cycles
(distinct-state validity reruns, EMA-dropout and prior-shooting confound
repairs, `research/cycles/hard_text/2026-07-17-gar-proposal-coverage.md`) are
closed; their open question — proposal coverage at fixed K=32 — is answered by
the 2026-08-13 report.

## Storage note (2026-08-13)

`/vol/home-vol2` hit 92%. `runs/autonomy/token_igsm` went 144 G -> 50 G:
the FAILED `2026-07-31-...-50k-s1-v8` feature cache was deleted (46.6 G; see
`source/FEATURES_DELETED.json` for the file list and regeneration route, splits
and pool retained), and 48 G of regenerable caches plus intermediate checkpoints
were moved to `/vol/tmp2/laitenbf/textjepa-archive/` (byte-verified; each
vacated location has an `ARCHIVED.json`). All models, `best.pt`, gates, splits
and the parameter-matched Qwen baselines were retained in place.
