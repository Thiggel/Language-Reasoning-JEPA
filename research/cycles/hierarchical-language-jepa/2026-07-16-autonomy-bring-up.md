# Cycle: bounded autonomy bring-up

Status: implementation; no experiments submitted

## Decision

Can the project execute one reproducible decision cycle across Grünau and the
three Slurm sites without duplicate writers, unbounded storage, mutable code,
or dependence on a long-lived Codex context?

## Why now

The repository already contains many waves and hundreds of run directories.
Cross-site execution has worked through bespoke scripts, but it lacks a single
durable state machine and compact validity contract. Scientific autonomy would
amplify those operational weaknesses.

## Work in this cycle

- Add a declarative plan and compact run-summary contract.
- Inventory Grünau by both memory and utilization; inventory Slurm sites by
  partitions and queues.
- Run every job from the exact planned Git commit.
- Refuse duplicate round IDs and storage-threshold violations.
- Retrieve compact artifacts, excluding checkpoints by default.
- Poll with an idempotent timer and start a fresh GPT-5.6 Sol, medium-reasoning
  Codex process only after a complete round is locally available.

## Validity gates

1. Controller/unit tests pass without network mutation.
2. Dry-run validation makes no scheduler call.
3. One manually reviewed seconds-scale Grünau run produces terminal markers
   and a compact summary.
4. Each Slurm backend separately passes snapshot, submit, poll, and retrieval.
5. Low-storage and failed-job tests pause new work.

## Result

Pending. No GPU job has been submitted by this cycle.

## Next step

Complete local validation, clean or isolate the checkout, and perform the
Grünau smoke test manually.

