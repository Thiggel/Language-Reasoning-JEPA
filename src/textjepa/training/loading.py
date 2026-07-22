"""DataLoader settings shared by high-throughput training entry points."""

from __future__ import annotations

import torch


def performance_loader_kwargs(
    workers: int, device: torch.device, *, persistent: bool = True,
) -> dict:
    """Use asynchronous host-to-device staging without invalid worker args."""
    workers = int(workers)
    result = {
        "pin_memory": device.type == "cuda",
        "persistent_workers": bool(persistent and workers > 0),
    }
    if workers > 0:
        result["prefetch_factor"] = 4
    return result
