# Grete (GWDG/NHR)

Verified live on 2026-07-16: SSH alias `Grete`, user `u27381`, account
`bem00089`, `$PROJECT=/projects/extern/nhr/nhr_be/bem00089/dir.project`, clone
root `$PROJECT/TextJEPA`, and shared environment `$PROJECT/babylm/.venv`.
Available Slurm families include `grete:shared`, `grete`, H100 variants, and
test/interactive partitions; the checked-in conservative default is one A100
in `grete:shared` with the `normal` QoS/account policy.

The official GWDG guide emphasizes scheduler-described resource requests and
shared storage. Always query live partitions and this account's association;
do not copy Lise or Alex directives blindly. Put persistent data under
`$PROJECT` and job-local temporary files under `$SLURM_TMPDIR`.

Primary source: https://docs.hpc.gwdg.de/start_here/index.html
