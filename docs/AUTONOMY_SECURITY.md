# Autonomy and budget boundary

`automation/config.toml` is human policy. Autonomous Codex must not edit it,
the controller, skills, AGENTS.md, the charter, Git configuration, credentials,
or scheduler policy. Keep the active policy copy outside the repository so a
workspace-write Codex process cannot expand its own authority.

The plan is data, not a shell script: commands are argv arrays; identifiers,
environment names, Slurm fields, walltime, GPUs, job count, and projected
GPU-hours are validated. Training code at an approved Git commit still runs
with the user's account and can perform arbitrary actions available to that
account. Review model-authored code and use cluster/project accounts with the
least privilege available.

Submission is idempotent at round level. A registered round ID cannot be
submitted again, and partial submission remains recorded for human review.
The controller lock prevents concurrent local writers. It cannot reserve a
Grünau GPU against other users; Slurm backends provide authoritative
reservations.

Always require human approval for multi-node jobs, paper-scale campaigns,
budget increases, new data transfers, destructive cleanup, credentials,
publication, and cancellation of unrelated jobs.

