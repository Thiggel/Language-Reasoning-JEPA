"""Deterministic sampling utilities for procedurally generated datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class FreshEpochSampler(Sampler[int]):
    """Yield a disjoint deterministic index block on every epoch.

    The procedural datasets key generation by ``(seed, index)``.  Offsetting
    indices by ``epoch * len(dataset)`` therefore gives fresh problems without
    mutating worker-local dataset copies (and remains deterministic with
    persistent DataLoader workers).  ``set_epoch`` also seeds the within-epoch
    permutation, so resumed or repeated runs reproduce exactly the same data.
    """

    def __init__(self, data_source, seed: int = 0, shuffle: bool = True):
        self.data_source = data_source
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=generator).tolist()
        else:
            order = range(n)
        offset = self.epoch * n
        return iter(offset + i for i in order)


class DistributedFreshEpochSampler(Sampler[int]):
    """Shard :class:`FreshEpochSampler` without repeating generated problems.

    Unlike ``DistributedSampler``, the yielded indices retain the epoch offset
    used by the procedural datasets.  ``drop_last=True`` is deliberate: scale
    runs choose a dataset size divisible by the number of ranks, so every rank
    performs the same number of optimizer collectives and the global exposure
    count remains exact.
    """

    def __init__(self, data_source, rank: int, world_size: int, seed: int = 0,
                 shuffle: bool = True):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if len(data_source) % world_size:
            raise ValueError("dataset size must be divisible by world_size")
        self.data_source = data_source
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        n = len(self.data_source)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(n, generator=generator)
        else:
            order = torch.arange(n)
        shard = order[self.rank::self.world_size]
        offset = self.epoch * n
        return iter((shard + offset).tolist())
