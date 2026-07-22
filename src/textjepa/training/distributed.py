"""Small native-DDP helpers used by long from-scratch scale runs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize(requested_device: str) -> DistributedContext:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        use_cuda = str(requested_device).startswith("cuda") and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        return DistributedContext(
            int(os.environ["RANK"]), local_rank, world,
            torch.device("cuda", local_rank) if use_cuda else torch.device("cpu"),
        )
    device = torch.device(requested_device)
    return DistributedContext(0, 0, 1, device)


def wrap(model: torch.nn.Module, context: DistributedContext):
    if context.world_size == 1:
        return model
    kwargs = {"broadcast_buffers": False}
    if context.device.type == "cuda":
        kwargs.update(
            device_ids=[context.local_rank], output_device=context.local_rank
        )
    return DistributedDataParallel(model, **kwargs)


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def close(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.destroy_process_group()
