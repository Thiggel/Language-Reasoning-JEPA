# Repository operating notes

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
