# Repository operating notes

## Scientific subprojects

- Use `projects/intent_phrase/`, `projects/token_igsm/`, and
  `projects/sequence_edit/` as the stable subproject entry points.
- Keep observed intent-phrase, token-level causal iGSM, and sequence-edit
  claims, status, experiment plans, and reports separate. Shared library code
  may remain under `src/textjepa/`.
- Preserve historical paths under `research/intent_phrase/`,
  `research/hard_text/`, `research/archive/edit_track/`, and `runs/`; do not
  move checkpoints or rename run families merely for cosmetic organization.
- New subproject-specific documents should link from the corresponding
  `projects/<name>/README.md` and state explicitly when an experiment uses
  oracle, symbolic, candidate-privileged, or cross-project information.

## Grünau GPU cluster

- The repository and its run artifacts live on the same shared filesystem on
  every Grünau server. The absolute project path is
  `/vol/home-vol2/ml/laitenbf/TextJEPA` everywhere, so jobs can be resumed or
  launched on another server without copying files.
- Run `gruenau-gpus` before scheduling GPU work. It queries Grünau 1--12 and
  labels a GPU free only when both memory use and utilization are below its
  configured thresholds. Do not infer availability from utilization alone:
  GPUs with large allocated memory and 0% instantaneous utilization are busy.
- Use `gruenau <N>` to log into server `N`, for example `gruenau 11`.
  Non-interactive automation may use the equivalent SSH host
  `laitenbf@gruenau<N>.informatik.hu-berlin.de`.
- Grünau 3--6 currently have no accessible `nvidia-smi`; treat them as having
  no schedulable GPU unless `gruenau-gpus` later reports otherwise.
- Typical hardware: Grünau 1 has two V100s and one RTX 6000; Grünau 2 has
  three RTX 6000s; Grünau 7--8 have four RTX A6000s each; Grünau 9--10 have
  three A100s each; Grünau 11 has four H100s; Grünau 12 has ten L40s.
- Coordinate through shared run completion/failure markers and use unique run
  names. Before launching remotely, check both GPU availability and whether
  the run is already active or complete to avoid duplicate writers.

## External Slurm clusters

- `ssh alex`, `ssh Lise`, and `ssh Grete` are independent filesystems. Clone
  the repository on each cluster and synchronize only intentional source and
  configuration changes; never assume Grünau checkpoints are visible there.
- Put repositories, caches, checkpoints, and generated data in the cluster's
  work/project filesystem: `$WORK/TextJEPA` on Alex and Lise, and
  `$PROJECT/TextJEPA` on Grete. Avoid large or high-inode artifacts in `$HOME`.
- Reuse one compatible shared environment rather than creating a venv per
  clone. The environments validated for this repository are `$WORK/.venv` on
  Alex, `~/linear-attention/.venv` on Lise, and `$PROJECT/babylm/.venv` on
  Grete. Set `PYTHONPATH=$TEXTJEPA_ROOT/src` instead of editable-installing the
  project into those shared environments.
- Alex requires `http_proxy=http://proxy:80` and the corresponding HTTPS
  variable for outbound access; its jobs load `cuda/12.8.1`. Lise A100 jobs
  load `sw.a100`. Follow each cluster's Slurm resource policy rather than
  copying directives blindly (for example, Alex rejects explicit `--mem` for
  GPU jobs).
- Use a job-specific `$SLURM_TMPDIR` when available, or a path containing
  `$SLURM_JOB_ID`; never share one multiprocessing temporary directory across
  array tasks.
- Current deployment roots and schedulers are encoded in
  `scripts/slurm_token_prior_overnight.sbatch`. Keep run names seed-qualified
  and use completion/failure markers so retries do not create duplicate
  checkpoint writers.

## Autonomous research contract

- Read `research/CHARTER.md`, `research/STATE.md`, `research/EVIDENCE.md`,
  `research/QUESTION_BACKLOG.md`, `research/EXPERIMENT_INDEX.md`, and the
  current cycle linked from `STATE.md` before directing a new experiment.
  Follow historical links only when relevant; do not ingest every raw log or
  wave into one context.
- Act as a skeptical research director. Choose one falsifiable decision at a
  time and optimize reliable information per elapsed hour. State in advance
  which outcomes change the direction, use the smallest faithful experiment,
  and scale only after the charter's validity gates pass.
- Challenge leakage, collapse, effective rank, capacity, optimization,
  target-encoder dynamics, masks and boundaries, data packing and mixture,
  trajectory length, within-batch diversity, proposal order, and metric
  validity. Compare against information-matched flat and negative controls.
- Tune competing stabilizers fairly with method-appropriate coefficient
  ranges. Add a learning-rate cross-check when loss scale or gradient flow
  changes materially. Do not run broad sweeps merely because GPUs are idle.
- Keep observed-intent, action-free, and hard-text claims separate. Label
  symbolic/oracle/candidate-privileged diagnostics. A successful process is
  not automatically a scientifically valid result.
- Record each decision cycle in one file under `research/cycles/<subproject>/`.
  Update the compact evidence, experiment, and decision ledgers; preserve
  failed and invalid runs with explicit reasons. Use intuitive full names in
  plans, tables, plots, and prose.
- Complete every oversight or result-analysis cycle with a self-contained
  bundle under `research/reports/<subproject>/<date>-<name>/`. Follow
  `research/templates/REPORT.md` and use `$explain-research`. Begin as if the
  reader barely knows what a neural network or language model is, then build
  toward ICLR-level technical and evidential detail. Include an intuitive
  figure, interpreted table, fairness discussion, limitations, glossary, and
  concrete steering questions. A cycle log or terse final message is not a
  substitute. `automation/validate_reports.py` must pass before the next plan.
- Read applicable human notes from `.researchctl/steering/inbox/` before
  selecting a decision. Explain how they affected the choice. Respect the
  configured unread-report limit; autonomy must stop admitting new rounds when
  it has advanced beyond the human's reviewed evidence.

## Controller boundary

- Use the project skills in `.agents/skills/` for research direction,
  experiment design, cluster orchestration, result analysis, literature
  review, and Beamer synthesis.
- Codex may implement and test the code needed for the current decision and
  write `research/NEXT_PLAN.json`. It must not directly call SSH, `sbatch`,
  `scancel`, or detached launch commands during an autonomous oversight turn.
  The deterministic interface is `automation/bin/researchctl`; operator usage
  is documented in `docs/AUTONOMOUS_RESEARCH.md`.
- Inventory before planning GPU work. Slurm owns allocation on Alex, Lise, and
  Grete. On Grünau, require both low allocated memory and low utilization and
  remember that direct observation cannot prevent a race with another user.
- Run experiments from exact Git snapshots. Keep caches, repositories, runs,
  and generated data on work/project storage and use job-specific temporary
  directories. Never put large artifacts in remote homes.
- Do not bypass `research/STOP`, storage or GPU-hour limits, protected policy,
  duplicate-round checks, or manual `--execute` submission. Humans approve
  paper-scale or multi-node work, budget increases, new dataset transfers,
  credentials, publication, destructive cleanup, and cancellation of jobs
  outside the controller.
- Fresh oversight processes use `gpt-5.6-sol` with medium reasoning. Keep the
  compact repository memory as continuity; do not depend on resuming an
  indefinitely growing conversation.
