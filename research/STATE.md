# Current scientific state

Last updated: 2026-07-16

## Active decision

The controller smoke has passed on Grünau after correcting validation-worker
configuration. The first scientific round asks whether hierarchy changes the
shared causal state toward future-relevant abstraction and whether duration-
weighted upper losses resolve gradient competition.

Current cycle:
[`cycles/hard_text/2026-07-16-hierarchy-gradient-abstraction.md`](cycles/hard_text/2026-07-16-hierarchy-gradient-abstraction.md)

## Established context

- The repository has three scientific subprojects with stable entry points
  under `projects/`: observed intent phrase, token-level causal iGSM, and
  sequence edit. Shared code does not imply shared claims.
- Observed-intent and hard-text tracks answer different questions; keep them
  separate.
- Corrected discrete hierarchy confirmations are negative relative to the flat
  controller. Earlier positive results were invalidated by proposal ordering
  and truncation artifacts.
- Faithful iGSM required stable problem-specific shuffled menus because the
  reference order leaked an always-first solution.
- Infrastructure is being made reproducible before another scientific wave:
  compact run summaries, immutable snapshots, cross-cluster inventory, and
  durable polling are the present engineering gate.

The intent-phrase subproject has a separate paper roadmap at
[`../projects/intent_phrase/PAPER_ROADMAP.md`](../projects/intent_phrase/PAPER_ROADMAP.md).
Its immediate paper decision is to localize why the causal J3 preference model
trails the matched token intent policy; hierarchy is excluded from that recipe.

## Next handoff

Bring up the controller in dry-run mode, then run one seconds-scale Grünau
smoke job. Do not enable fresh Codex wake-ups or automatic submission while the
main checkout is dirty. After backend smoke tests, the first scientific cycle
should validate hierarchy-health measurements on a tiny controlled setting,
not launch a broad sweep.
