"""Comparable FLOP and wall-time accounting for staged language planning.

FLOPs are estimates unless a caller supplies a measured hardware counter.
The default parameter-token convention is explicit so fractions remain
comparable across additions:

* frozen inference: ``2 * parameters * processed_tokens``;
* trainable forward+backward: ``6 * parameters * processed_items``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator


@dataclass
class ComponentCompute:
    estimated_flops: float = 0.0
    wall_seconds: float = 0.0
    calls: int = 0
    items: int = 0


class ComputeLedger:
    def __init__(self) -> None:
        self.components: dict[str, ComponentCompute] = {}

    def add(
        self,
        name: str,
        *,
        estimated_flops: float = 0.0,
        wall_seconds: float = 0.0,
        calls: int = 1,
        items: int = 0,
    ) -> None:
        if estimated_flops < 0 or wall_seconds < 0 or calls < 0 or items < 0:
            raise ValueError("compute measurements must be nonnegative")
        row = self.components.setdefault(name, ComponentCompute())
        row.estimated_flops += float(estimated_flops)
        row.wall_seconds += float(wall_seconds)
        row.calls += int(calls)
        row.items += int(items)

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        estimated_flops: float = 0.0,
        items: int = 0,
    ) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.add(
                name,
                estimated_flops=estimated_flops,
                wall_seconds=perf_counter() - start,
                items=items,
            )

    def summary(self) -> dict:
        total_flops = sum(
            row.estimated_flops for row in self.components.values()
        )
        total_seconds = sum(
            row.wall_seconds for row in self.components.values()
        )
        return {
            "flop_convention": {
                "frozen_inference_per_parameter_token": 2,
                "train_forward_backward_per_parameter_item": 6,
                "kind": "estimated",
            },
            "total_estimated_flops": total_flops,
            "total_component_wall_seconds": total_seconds,
            "components": {
                name: {
                    "estimated_flops": row.estimated_flops,
                    "flop_fraction": (
                        row.estimated_flops / total_flops
                        if total_flops else 0.0
                    ),
                    "wall_seconds": row.wall_seconds,
                    "wall_fraction": (
                        row.wall_seconds / total_seconds
                        if total_seconds else 0.0
                    ),
                    "calls": row.calls,
                    "items": row.items,
                }
                for name, row in sorted(self.components.items())
            },
        }


def parameter_count(module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def inference_flops(parameters: int, processed_tokens: int) -> float:
    if parameters < 0 or processed_tokens < 0:
        raise ValueError("FLOP inputs must be nonnegative")
    return float(2 * parameters * processed_tokens)


def training_flops(parameters: int, processed_items: int) -> float:
    if parameters < 0 or processed_items < 0:
        raise ValueError("FLOP inputs must be nonnegative")
    return float(6 * parameters * processed_items)
