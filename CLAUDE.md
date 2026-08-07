# CLAUDE.md — working notes for this repo

Read together with `AGENTS.md` (cluster hosts, notation, subproject boundaries)
and `docs/clusters/*.md` (per-cluster ssh/Slurm details).

## Cluster scheduling tips (from the project owner)

- **Lise**: Slurm queue times are often long — jobs can sit pending for hours.
  Don't plan tight iteration loops around Lise; use it for wide, non-urgent
  sweeps.
- **Alex**: queueing can also take a while, though usually less than Lise.
- **Grünau**: no mandatory Slurm; GPUs are grabbed directly. Especially at
  night and in the morning almost everything can be free — it often makes
  sense to move jobs from the Slurm clusters to Grünau then. Always check
  actual availability first (both memory *and* utilization; allocated memory
  with 0% utilization means busy).
- Grünau shares this filesystem; Alex/Lise/Grete have their own — results
  produced there must be synced back explicitly.

## Operational conventions

- Experiments launch directly over SSH/Slurm (the old researchctl automation
  was removed 2026-08-06). Keep the run-directory convention:
  `runs/autonomy/<project>/<round-id>/<cell>/` with `job.sh`, `state`,
  `stdout.log`, `stderr.log`, `exit_code`, result JSONs; run from an immutable
  snapshot under `runs/autonomy/_code/<sha>` (`git archive`).
- Training python: `.venv/bin/python` (do not import matplotlib there);
  figures/LaTeX/eval helpers: `.venv2/bin/python`.
- All paper writing lives outside this repo in
  `/vol/home-vol2/ml/laitenbf/TextJEPA-paper/` (own git repo; `overleaf/` is
  the Overleaf project).

## Handoff discipline (required)

- Maintain `projects/intent_phrase/CAMPAIGN_LOG.md` as the living handoff:
  current frozen recipe + headline numbers, what is running where, last
  decisions, and next planned steps. Update it after every experiment round
  or decision; never rely on chat context alone to carry project state.
- Keep it short: when a stage completes, compress its entries to a few lines
  and move the detail into a dated report under
  `research/reports/intent_phrase/<date>-<topic>/REPORT.md` (mirrored to
  `/vol/home-vol2/ml/laitenbf/TextJEPA-paper/reports/`).
- `HANDOFF.md` and `RESULTS.md` at the repo root are frozen 2026-07 history;
  do not extend them.

## Communication preferences (project owner)

- Explain findings in plain, intuitive, concise English; no invented jargon.
  Briefly re-explain what each experiment does and what it showed, every time
  it is mentioned.
- Every decision should be weighed against one goal: a 100% reviewer-proof
  ICLR paper. Label oracle/symbolic/candidate-privileged evidence explicitly;
  do not invent new toy problems — stay in the established environments.
