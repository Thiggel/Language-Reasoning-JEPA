---
name: cluster-orchestrator
description: Inventory and safely place TextJEPA jobs across Grünau, Alex, Lise, and Grete through the repository research controller. Use when checking GPU capacity, validating or submitting a run plan, polling jobs, retrieving results, diagnosing backend/storage state, or installing the durable watcher.
---

# Orchestrate bounded cluster work

Read `docs/AUTONOMOUS_RESEARCH.md`, `docs/AUTONOMY_SECURITY.md`, and only the
relevant file under `docs/clusters/`. Use `automation/bin/researchctl`; do not
reimplement SSH, Slurm, locks, storage guards, snapshots, or retrieval in an
ad-hoc shell script.

Run `inventory` before placement and `storage` before a large round. Grünau is
free only when both memory allocation and utilization pass the thresholds.
Treat Slurm as the reservation authority on Alex, Lise, and Grete; use queue and
partition inventory to estimate time-to-decision, not to address GPUs directly.

Validate and finalize every plan. Submission is a dry run unless a human or the
trusted outer workflow explicitly authorizes `--execute`. Keep round/run names
unique and seed-qualified. Use exact Git snapshots, shared environments,
work/project caches, and job-specific temporary directories.

When a guard fires, preserve state and report the exact condition. Do not delete
runs, increase budgets, change accounts/partitions, cancel jobs, or bypass the
STOP file without human direction.

