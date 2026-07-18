# Operating the TextJEPA research loop

The interface is `automation/bin/researchctl`. Codex designs and documents one
decision cycle; `researchctl` is the durable authority for inventory, storage
checks, immutable snapshots, submission, polling, retrieval, and wake-ups.

For the human reading interface, unread-report guard, and regular steering
routine, see
[`RESEARCH_READER_AND_STEERING.md`](RESEARCH_READER_AND_STEERING.md).

## What runs where

Grünau uses direct detached jobs because its GPUs are not authoritatively
reserved by the helper. The controller rechecks every GPU with `nvidia-smi`
and calls it free only when both allocated memory and utilization are below the
configured thresholds. It records its own reservations, but an unrelated user
can still race for the same GPU. Prefer Grünau Slurm if it becomes reliably
available.

Alex, Lise, and Grete use Slurm. “Free GPU” there means available scheduler
capacity, not a GPU that should be addressed directly. Inventory captures
partitions, node states, total GPU requests, and this account's queue; Slurm
owns placement and exclusion.

Every job runs from `_snapshots/<full-git-sha>`, never from the mutable checkout.
External snapshots are streamed from `git archive`; no remote branch reset is
needed. Shared Python environments and `PYTHONPATH=<snapshot>/src` avoid a venv
per snapshot.

## Daily interaction

```bash
# Read-only overview
automation/bin/researchctl status
automation/bin/researchctl inventory
automation/bin/researchctl storage

# Validate a Codex- or human-authored plan (no submission)
cp automation/examples/round-plan.json research/NEXT_PLAN.json
$EDITOR research/NEXT_PLAN.json
automation/bin/researchctl validate-plan
automation/bin/researchctl finalize-plan
automation/bin/researchctl submit-plan .researchctl/plans/<round>.resolved.json

# Actual scheduler mutation: always explicit
automation/bin/researchctl submit-plan \
  .researchctl/plans/<round>.resolved.json --execute
automation/bin/researchctl refresh
automation/bin/researchctl status
```

Ask Codex normally in the repository for interactive research work, for
example: “Use `$research-director` to audit the latest completed round and
design the smallest next decision cycle. Do not submit it.” Review the cycle
document and plan, then submit through the controller.

For a fresh non-interactive oversight process:

```bash
automation/bin/researchctl wake
```

This is initially disabled. Copy the configuration outside the repository,
set `enabled = true`, export `RESEARCH_CONFIG`, and use a clean dedicated
checkout before enabling it. The command uses GPT-5.6 Sol with medium reasoning
and stores JSONL plus the final response under `.researchctl/oversight/`. After
Codex exits, the controller rejects protected, oversized, or over-broad changes,
runs the configured verification command, creates the exact Git commit, and
resolves `git_commit: AUTO` in an ignored controller-side plan copy. Codex does
not commit or submit directly.

On this machine, `cs` is an interactive alias for `codex --search
--dangerously-bypass-approvals-and-sandbox`; it is not an executable available
to systemd. The controller therefore calls the real `codex` binary with
`--search`, `workspace-write`, and `approval_policy="never"`. This preserves
the requested search-enabled GPT-5.6 Sol/medium behavior without inheriting the
alias's full-access bypass.

Dedicated project worktrees are clean by default. For unattended recovery from
a failed verification gate, an operator may set
`resume_dirty_project_worktrees = true` under `[codex]`. The next wake may then
continue bounded changes owned by that project. Protected controller paths,
sibling-project memory, and file/byte limits still block the wake; this option
is therefore not a general bypass for arbitrary dirty worktrees.

## Watcher choices

For a durable watcher, install the user timer:

```bash
automation/bin/install-user-timer.sh
systemctl --user list-timers textjepa-research-watch.timer
journalctl --user -u textjepa-research-watch.service -n 100
```

The timer calls one idempotent `tick` every two minutes. For a foreground
debugging loop, use `automation/bin/researchctl watch --interval 60`. The timer
is preferable for normal operation because it survives terminal disconnects.
Codex Scheduled tasks can also revisit local projects when the desktop app and
machine are running, but they are not the source of truth for multi-cluster
jobs or remote artifact retrieval.

## Safe bring-up

1. Copy `automation/config.example.toml` to
   `~/.config/textjepa-research/controller.toml`, set mode 600, and export
   `RESEARCH_CONFIG`.
2. Run `init`, `doctor`, `inventory`, `storage`, and `status`.
3. Keep Codex disabled. Dry-run the example plan.
4. Submit one ten-step/seconds-scale Grünau smoke run after inspecting the
   generated plan.
5. Test one Slurm site at a time. Verify immutable snapshot, job ID, terminal
   accounting, and compact retrieval.
6. Deliberately test a failed command and a too-low storage threshold.
7. Use a clean dedicated checkout, then enable Codex wake while retaining
   manual plan submission.
8. Consider `auto_submit_after_wake = true` only after several reviewed
   scientific cycles. This completes the autonomous chain, but the same plan,
   storage, active-job, rolling GPU-hour, protected-path, and STOP guards still
   apply.

For unattended Codex, use a dedicated clean checkout after committing this
package and the intended scientific baseline:

```bash
git worktree add ../TextJEPA-autonomy -b autonomy/research-loop HEAD
cd ../TextJEPA-autonomy
cp automation/config.example.toml \
  ~/.config/textjepa-research/controller.toml
# Edit [project].root to this worktree, then:
export RESEARCH_CONFIG=~/.config/textjepa-research/controller.toml
automation/bin/researchctl doctor
```

Do not point autonomous wake at the current working checkout while it contains
uncommitted human work.

## Emergency controls

```bash
touch research/STOP
automation/bin/researchctl pause "investigating unexpected behavior"
systemctl --user disable --now textjepa-research-watch.timer
```

These prevent new controller actions. They do not cancel accepted jobs. Use
the appropriate explicit `scancel` or remote process termination only after
identifying the exact job. Remove `research/STOP` and run `resume` when safe.

## Storage behavior

Repositories, caches, temporary files, and outputs are rooted in Grünau's
shared volume, Alex/Lise work storage, or Grete project storage—not remote
homes. Before submission the controller checks free bytes and used percentage
on both local and target filesystems. It retrieves compact summaries, metrics,
logs, tables, and figures up to the configured size, excluding checkpoints.

The controller never silently deletes scientific runs or checkpoints. When a
guard fires, inspect with `researchctl storage`, archive or remove artifacts
deliberately, then resume. Keep caches shared per cluster and use
`$SLURM_TMPDIR` for job-local high-inode temporary data.
