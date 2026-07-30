"""Legacy pre-pivot counterfactual and planner-replay containers.

New staged jobs use :mod:`textjepa.data.language_planning` and its global
prefix ``input_ids`` contract. These types remain loadable for historical
artifacts but are not accepted as official dense-training examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Iterable

import torch


class CandidateSource(str, Enum):
    OBSERVED = "observed"
    GREEDY = "greedy"
    LOW_TEMPERATURE = "low_temperature"
    MEDIUM_TEMPERATURE = "medium_temperature"
    HIGH_TEMPERATURE = "high_temperature"
    CEM = "cem"
    PLANNING_FAILURE = "planning_failure"
    RANDOM_PRIOR = "random_prior"
    WORKER_ACHIEVED = "worker_achieved"


@dataclass(frozen=True)
class InformationProvenance:
    """Explicitly label information unavailable to normal inference."""

    oracle_terminal: bool = False
    symbolic: bool = False
    candidate_privileged: bool = False
    cross_project: bool = False


@dataclass(frozen=True)
class BoundaryPolicy:
    min_span: int = 1
    max_span: int = 128

    def normalize(self, boundaries: Iterable[int], sequence_length: int) -> list[int]:
        """Split long spans and merge too-short interior spans deterministically."""
        if sequence_length < 1 or self.min_span < 1 or self.max_span < self.min_span:
            raise ValueError("invalid boundary policy")
        original = [int(value) for value in boundaries]
        if any(value < 0 or value >= sequence_length for value in original):
            raise ValueError("boundary lies outside sequence")
        points = sorted(set(original))
        if not points or points[0] != 0:
            points.insert(0, 0)
        if points[-1] != sequence_length - 1:
            points.append(sequence_length - 1)
        merged = [points[0]]
        for point in points[1:-1]:
            if point - merged[-1] >= self.min_span:
                merged.append(point)
        if points[-1] - merged[-1] < self.min_span and len(merged) > 1:
            merged.pop()
        merged.append(points[-1])
        split = [merged[0]]
        for point in merged[1:]:
            while point - split[-1] > self.max_span:
                split.append(split[-1] + self.max_span)
            if point != split[-1]:
                split.append(point)
        return split


@dataclass
class CounterfactualBatch:
    """Root-by-candidate hidden-state branches before rectangular flattening."""

    token_ids: torch.Tensor
    hidden_states: torch.Tensor
    lengths: torch.Tensor
    root_ids: torch.Tensor
    temperatures: torch.Tensor

    def validate(self) -> None:
        if self.token_ids.ndim != 3:
            raise ValueError("token_ids must be [roots, candidates, length]")
        if self.hidden_states.shape[:3] != self.token_ids.shape:
            raise ValueError("hidden states must align with every branch token")
        if self.lengths.shape != self.token_ids.shape[:2]:
            raise ValueError("lengths must be [roots, candidates]")
        if bool((self.lengths < 1).any()) or bool(
            (self.lengths > self.token_ids.shape[-1]).any()
        ):
            raise ValueError("invalid branch length")
        if self.root_ids.shape != self.token_ids.shape[:2]:
            raise ValueError("one root id is required per branch")
        if self.temperatures.shape != self.token_ids.shape[:2]:
            raise ValueError(
                "temperature must contain one scalar per root/candidate pair"
            )

    def flatten(self) -> dict[str, torch.Tensor]:
        """``[R,M,L,*] -> [RM,L,*]`` for ordinary learner batches."""
        self.validate()
        roots, candidates, length = self.token_ids.shape
        flat_lengths = self.lengths.reshape(-1)
        flat_boundaries = torch.stack([
            torch.zeros_like(flat_lengths), flat_lengths
        ], -1)
        width_mask = torch.arange(
            length, device=self.token_ids.device
        )[None] < flat_lengths[:, None]
        return {
            "token_ids": self.token_ids.reshape(roots * candidates, length),
            "hidden_states": self.hidden_states.reshape(
                roots * candidates, length, self.hidden_states.shape[-1]
            ),
            "lengths": flat_lengths,
            "root_ids": self.root_ids.reshape(-1),
            "temperatures": self.temperatures.reshape(-1),
            "attention_mask": width_mask,
            "boundaries": flat_boundaries,
        }


@dataclass(frozen=True)
class PlannerReplayRecord:
    """Candidate-level audit data separating model, geometry, and execution."""

    problem_id: str
    root_id: str
    source: str
    action_tokens: list[int]
    predicted_endpoint_cost: float
    exact_endpoint_cost: float
    achieved_endpoint_cost: float | None
    action_log_probability: float
    valid_symbolic_step: bool | None
    symbolic_distance_before: float | None
    symbolic_distance_after: float | None
    cem_iteration: int | None
    wall_seconds: float
    predictor_flops: float
    provenance: InformationProvenance

    def validate(self) -> None:
        if not self.problem_id or not self.root_id or not self.action_tokens:
            raise ValueError("replay record requires problem, root, and action")
        if self.wall_seconds < 0 or self.predictor_flops < 0:
            raise ValueError("compute measurements cannot be negative")
        try:
            CandidateSource(self.source)
        except ValueError as error:
            raise ValueError(f"unknown candidate source: {self.source}") from error
        numeric = (
            self.predicted_endpoint_cost, self.exact_endpoint_cost,
            self.action_log_probability, self.wall_seconds,
            self.predictor_flops,
        )
        if self.achieved_endpoint_cost is not None:
            numeric += (self.achieved_endpoint_cost,)
        if not all(torch.isfinite(torch.tensor(value)) for value in numeric):
            raise ValueError("replay numeric fields must be finite")
        if self.cem_iteration is not None and self.cem_iteration < 0:
            raise ValueError("CEM iteration must be nonnegative")
        symbolic = (
            self.valid_symbolic_step,
            self.symbolic_distance_before,
            self.symbolic_distance_after,
        )
        if any(value is not None for value in symbolic) and not all(
            value is not None for value in symbolic
        ):
            raise ValueError("symbolic replay fields must be supplied together")


def write_replay_jsonl(path: str | Path, records: Iterable[PlannerReplayRecord]) -> None:
    """Append replay metadata without mutating or consolidating legacy logs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            record.validate()
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_replay_jsonl(path: str | Path) -> list[PlannerReplayRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["provenance"] = InformationProvenance(**payload["provenance"])
            record = PlannerReplayRecord(**payload)
            record.validate()
            records.append(record)
    return records
