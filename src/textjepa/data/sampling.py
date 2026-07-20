"""Deterministic sampling utilities for procedurally generated datasets."""

from __future__ import annotations

import torch
from torch.utils.data import Sampler


class GroupedTrajectoryBatchSampler(Sampler[list[int]]):
    """Put every trajectory variant of each sampled problem in one batch."""

    def __init__(self, base_size: int, variants: int, bases_per_batch: int,
                 seed: int = 0, fresh_per_epoch: bool = True):
        self.base_size = int(base_size)
        self.variants = int(variants)
        self.bases_per_batch = int(bases_per_batch)
        self.seed = int(seed)
        self.fresh_per_epoch = bool(fresh_per_epoch)
        self.epoch = 0
        if min(self.base_size, self.variants, self.bases_per_batch) < 1:
            raise ValueError("grouped trajectory sampler sizes must be positive")

    def __len__(self) -> int:
        return self.base_size // self.bases_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.base_size, generator=generator).tolist()
        offset = self.epoch * self.base_size if self.fresh_per_epoch else 0
        stop = len(self) * self.bases_per_batch
        for start in range(0, stop, self.bases_per_batch):
            batch = []
            for base in order[start:start + self.bases_per_batch]:
                first = (offset + base) * self.variants
                batch.extend(first + variant for variant in range(self.variants))
            yield batch


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
