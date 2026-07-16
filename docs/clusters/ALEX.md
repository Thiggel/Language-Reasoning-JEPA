# Alex (NHR@FAU)

Verified live on 2026-07-16: SSH alias `alex`, account `c107fa`, clone root
`/home/atuin/c107fa/c107fa12/TextJEPA`, shared environment
`/home/atuin/c107fa/c107fa12/.venv`, and Slurm partitions including `a40`,
`a100`, `a100mig`, and `rtxpro6k`. Jobs load CUDA 12.8.1 and require the
documented outbound proxy when network access is needed.

Alex's official documentation describes eight-GPU A40 and A100 nodes,
exclusive requested GPUs, a 24-hour single-node limit, `--gres` GPU requests,
and node-local `$TMPDIR`. It also states that GPU shares determine CPU/memory,
so this project does not add an explicit `--mem` request. Multi-node access is
request-only and remains human-gated.

Primary source: https://doc.nhr.fau.de/clusters/alex/

