"""Search bookkeeping for intent-planning depth localization audits.

The functions here deliberately know nothing about neural models.  They make
the global-versus-root-balanced pruning intervention testable without granting
the deployed scorer access to exact action labels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from textjepa.data.igsm.graph import Problem
from textjepa.planning.search import _feasible


@dataclass(frozen=True)
class BeamLevel:
    depth: int
    candidates: int
    survivors: int
    surviving_roots: tuple[int, ...]
    optimal_root_survives: bool


@dataclass(frozen=True)
class BeamDecision:
    sequence: tuple[int | None, ...]
    score: float
    levels: tuple[BeamLevel, ...]


def expand_symbolic_prefixes(
    problem: Problem,
    resolved: frozenset[int],
    prefixes: list[list[int | None]],
) -> list[list[int | None]]:
    """Expand one level using the labeled symbolic feasible-action tree."""

    expanded: list[list[int | None]] = []
    for prefix in prefixes:
        reached = resolved | {action for action in prefix if action is not None}
        if problem.query in reached:
            expanded.append(prefix + [None])
            continue
        actions = _feasible(problem, frozenset(reached))
        expanded.extend(prefix + [action] for action in actions)
    return expanded


def prune_beam(
    sequences: list[list[int | None]],
    scores: np.ndarray,
    width: int,
    strategy: str,
) -> list[list[int | None]]:
    """Prune globally or divide one total budget across first actions."""

    score = np.asarray(scores, dtype=np.float64)
    if len(sequences) != len(score) or not sequences:
        raise ValueError("sequences and scores must be non-empty and aligned")
    if width < 1:
        raise ValueError("beam width must be positive")
    if strategy == "global":
        keep = np.argsort(score, kind="stable")[: min(width, len(score))]
        return [sequences[int(index)] for index in keep]
    if strategy != "root_balanced":
        raise ValueError(f"unknown pruning strategy: {strategy}")

    roots = list(dict.fromkeys(int(sequence[0]) for sequence in sequences))
    total = min(max(width, len(roots)), len(sequences))
    base, extra = divmod(total, len(roots))
    selected: list[int] = []
    for root_index, root in enumerate(roots):
        indices = np.asarray([
            index for index, sequence in enumerate(sequences)
            if sequence[0] == root
        ])
        allocation = min(base + int(root_index < extra), len(indices))
        local = indices[np.argsort(score[indices], kind="stable")[:allocation]]
        selected.extend(map(int, local))

    # A root may have fewer candidates than its allocation. Fill unused slots
    # globally without duplicating already retained prefixes.
    if len(selected) < total:
        chosen = set(selected)
        for index in np.argsort(score, kind="stable"):
            index = int(index)
            if index not in chosen:
                selected.append(index)
                chosen.add(index)
            if len(selected) == total:
                break
    selected.sort(key=lambda index: (score[index], index))
    return [sequences[index] for index in selected]


def search_symbolic_tree(
    problem: Problem,
    resolved: frozenset[int],
    depth: int,
    width: int,
    strategy: str,
    score: Callable[[list[list[int | None]]], np.ndarray],
) -> BeamDecision:
    """Run a scored beam while recording survival of task-useful roots."""

    if depth < 1:
        raise ValueError("search depth must be positive")
    roots = _feasible(problem, resolved)
    if not roots:
        return BeamDecision((None,), 0.0, ())
    optimal_roots = set(roots) & set(problem.query_ancestors)
    beam = [[root] for root in roots]
    levels: list[BeamLevel] = []
    for level in range(1, depth + 1):
        if level > 1:
            beam = expand_symbolic_prefixes(problem, resolved, beam)
        if not beam:
            raise RuntimeError("symbolic beam became empty")
        scores = np.asarray(score(beam), dtype=np.float64)
        candidates = len(beam)
        beam = prune_beam(beam, scores, width, strategy)
        surviving = tuple(sorted({int(sequence[0]) for sequence in beam}))
        levels.append(BeamLevel(
            depth=level,
            candidates=candidates,
            survivors=len(beam),
            surviving_roots=surviving,
            optimal_root_survives=bool(optimal_roots & set(surviving)),
        ))
    final_scores = np.asarray(score(beam), dtype=np.float64)
    selected = int(np.argmin(final_scores))
    return BeamDecision(
        tuple(beam[selected]), float(final_scores[selected]), tuple(levels)
    )
