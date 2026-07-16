# Grünau

Project and runs live at `/vol/home-vol2/ml/laitenbf/TextJEPA` on every server.
The interactive `gruenau-gpus` helper is useful to humans, but automation uses
direct SSH plus the same two-dimensional free rule: low allocated memory and
low utilization. A GPU with large memory allocation and 0% instantaneous
utilization is busy.

Schedulable inventory as of 2026-07-16: Grünau 1 (2×V100 32 GB, RTX 6000), 2
(3×RTX 6000), 7–8 (4×RTX A6000), 9–10 (3×A100 80 GB), 11 (H100 NVL/PCIe), and
12 (10×L40). Servers 3–6 are excluded until their `nvidia-smi` probes recover.

Detached direct placement is best-effort rather than an authoritative
reservation. The controller records its placements and uses unique run
directories, but another user can race it. Prefer Slurm when the local
installation can reliably schedule these GPUs.

