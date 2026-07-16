# Three-project assisted-autonomy architecture

## Outcome

TextJEPA now has one deterministic controller and watcher, but three isolated scientific directors: `intent_phrase`, `token_igsm`, and `sequence_edit`. Existing controller rounds remain at their original identifiers and paths. New plans use schema version 2, require a project, and write under `runs/autonomy/<project>/<round>/<job>/`.

## Stable project registry

Each `projects/<slug>/controller.toml` declares its scientific title, compact memory and report roots, current cycle, plan, branch/worktree, ownership patterns, prompt, activity flags, and per-project budgets. Directors initially read only shared charter/evidence plus their own compact memory, current cycle, steering, terminal summaries, and allocation snapshot.

Historical paths remain authoritative: `research/intent_phrase/`, `research/hard_text/`, `research/sequence_edit/`, `research/archive/edit_track/`, and existing `runs/` are not moved. Cross-project findings must be promoted explicitly to `research/SHARED_EVIDENCE.md`.

## State migration and compatibility

The version-2 migration annotates legacy rounds with a conservative project and `legacy=true`. It preserves round/job keys, backend IDs, commits, states, and local/remote paths. Ambiguous rounds remain `legacy/unclassified`. The command locks the controller, creates a checksummed backup, and replaces state atomically.

```bash
automation/bin/researchctl migrate-state --dry-run
automation/bin/researchctl migrate-state --execute
automation/bin/researchctl migrate-state --execute  # idempotent no-op
```

Rollback changes only controller metadata; it does not signal accepted jobs:

```bash
automation/bin/researchctl pause "controller metadata rollback"
touch research/STOP
automation/bin/researchctl migrate-state --rollback \
  .researchctl/migrations/<timestamp>/state.before-v2.json
```

Never remove original run directories or snapshots during rollback. Refresh observes legacy and project-qualified rounds through the same compatibility path.

## Fair future admissions

Every active project has a one-GPU guarantee when compatible capacity exists. Project manifests cap active GPUs, pending Slurm jobs, per-round GPU-hours, and seven-day GPU-hours. Legacy active jobs count toward current usage. If another project has a resolved runnable plan waiting, a project is held near 40% of visible/global slots. Idle capacity may be borrowed; nothing running is preempted.

```bash
automation/bin/researchctl projects
automation/bin/researchctl allocation
automation/bin/researchctl status --all
automation/bin/researchctl status --project token_igsm
```

The allocation output explains active and pending usage, guarantee, borrowing, remaining budgets, and whether a plan waits. Global storage, job, round, and weekly limits remain in force.

## Plans and manual approval

New plans require `schema_version: 2` and `project`. Each director writes its own plan path.

```bash
automation/bin/researchctl validate-plan --project token_igsm
automation/bin/researchctl finalize-plan --project token_igsm
automation/bin/researchctl submit-plan .researchctl/plans/<round>.resolved.json
# Only after human review:
automation/bin/researchctl submit-plan .researchctl/plans/<round>.resolved.json --execute
```

`auto_submit_after_wake` remains false. Enabling it later requires editing the external policy deliberately and setting the corresponding project manifest authorization; do this only after repeated reviewed cycles.

## Worktrees and integration

The three branches live in clean worktrees under `/vol/home-vol2/ml/laitenbf/TextJEPA-worktrees/`. Fresh Codex processes run with `gpt-5.6-sol`, medium reasoning, search, workspace-write, and no approval bypass. A director may commit on its branch after report/test/protected-path validation. Shared-source commits are integrated deliberately on the main integration branch; project branches must then merge or rebase that integration commit before relying on it.

Jobs can use any preserved exact branch commit. A job plan records the full commit and the controller creates an immutable archive snapshot, so integration does not mutate a running job.

## Watcher

Exactly one user systemd timer invokes an idempotent tick every two minutes. A tick refreshes and retrieves existing jobs even while admissions are paused. Under `STOP` or pause it remains observation-only. Terminal rounds wake only their classified project director. The unread-report limit prevents the system from getting ahead of the human. A non-blocking lock prevents overlapping ticks.

```bash
systemctl --user status textjepa-research-watch.timer
systemctl --user list-timers textjepa-research-watch.timer
journalctl --user -u textjepa-research-watch.service -n 100
```

Pause without affecting accepted jobs:

```bash
touch research/STOP
automation/bin/researchctl pause "human steering checkpoint"
```

Resume only after checks:

```bash
rm research/STOP
automation/bin/researchctl resume
```

## Research Reader and normal operation

See [RESEARCH_READER_AND_STEERING.md](RESEARCH_READER_AND_STEERING.md). Reports are read oldest-first, project filters isolate the three streams, and content-hash receipts release the human-review guard. Steering is written to `.researchctl/steering/inbox/<project>/`; a director must explain how it changed the next decision. The controller never marks a report read for the human.

## Live-job preservation evidence

The migration snapshot is `.researchctl/migrations/20260716T093805Z/`. The executed migration backup and the 48-row exact preservation comparison are under `.researchctl/migrations/20260716T094454Z/`. Each row verifies the same round/job key, backend ID, commit, state, local path, and remote path before and after migration.

## Operator cutover checklist

1. Keep admission paused while validation has any failure.
2. Run compilation, pytest, report/TOML/JSON/schema checks, shell checks, state dry-run/idempotence, status/allocation, UI indexing/API/browser checks, and an observation-only tick.
3. Compare live jobs again immediately before cutover.
4. Remove `research/STOP`, resume, and start only the user timer.
5. Run one tick and confirm no submission occurs.
6. Leave `auto_submit_after_wake=false` until the human explicitly changes the autonomy level.
