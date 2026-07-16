# Lise (NHR@ZIB)

Verified live on 2026-07-16: SSH alias `Lise`, user `bemflait`, account
`bem00089`, `$WORK=/scratch/usr/bemflait`, clone root `$WORK/TextJEPA`, shared
environment `~/linear-attention/.venv`, and A100 partitions
`gpu-a100:shared`, `gpu-a100`, and `gpu-a100:test`. Jobs load `sw.a100`.

Keep code, caches, runs, and generated data on scratch rather than `$HOME`.
Use `$SLURM_TMPDIR` or a job-ID-qualified directory for multiprocessing
temporary data. Confirm current partition/QoS policy with live `sinfo` and
`sacctmgr` before changing the controller configuration.

Primary source: https://nhr-zib.atlassian.net/wiki/spaces/PUB/pages/428112/Usage+Guide

