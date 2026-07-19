"""Exact terminal-privileged diagnostics for nested token-edit buffers."""

from __future__ import annotations


BOUNDARY = -1  # vocabulary ids are non-negative


def boundary_token_sequence(buffer: list[list[int]]) -> list[int]:
    """Flatten a buffer while retaining every between-step boundary."""
    sequence: list[int] = []
    for index, sentence in enumerate(buffer):
        if index:
            sequence.append(BOUNDARY)
        sequence.extend(int(token) for token in sentence)
    return sequence


def token_levenshtein(
    left: list[int], right: list[int], max_distance: int | None = None
) -> int:
    """Exact unit-cost Levenshtein distance using linear memory.

    ``max_distance`` enables an exact diagonal band when a known edit script
    supplies an upper bound, as faithful corruption trajectories do.
    """
    if len(left) > len(right):
        left, right = right, left
    if max_distance is not None:
        bound = max(0, int(max_distance))
        if len(right) - len(left) > bound:
            return bound + 1
        infinity = bound + 1
        previous = {column: column for column in range(min(len(left), bound) + 1)}
        for row, right_token in enumerate(right, start=1):
            start = max(0, row - bound)
            stop = min(len(left), row + bound)
            current: dict[int, int] = {}
            for column in range(start, stop + 1):
                if column == 0:
                    current[column] = row
                    continue
                current[column] = min(
                    current.get(column - 1, infinity) + 1,
                    previous.get(column, infinity) + 1,
                    previous.get(column - 1, infinity)
                    + int(left[column - 1] != right_token),
                )
            previous = current
        return previous.get(len(left), infinity)
    previous = list(range(len(left) + 1))
    for row, right_token in enumerate(right, start=1):
        current = [row]
        for column, left_token in enumerate(left, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(left_token != right_token),
            ))
        previous = current
    return previous[-1]


def boundary_token_edit_distance(
    buffer: list[list[int]], target: list[list[int]],
    max_distance: int | None = None,
) -> int:
    return token_levenshtein(
        boundary_token_sequence(buffer), boundary_token_sequence(target),
        max_distance=max_distance,
    )


def exact_one_step_advantage(
    before: list[list[int]], after: list[list[int]], target: list[list[int]],
    max_distance: int | None = None,
) -> int:
    """Positive iff the literal edit reduces exact terminal distance."""
    return (
        boundary_token_edit_distance(before, target, max_distance)
        - boundary_token_edit_distance(
            after, target,
            None if max_distance is None else max_distance + 1,
        )
    )


def exact_one_step_advantages(
    before: list[list[int]], outcomes: list[list[list[int]]],
    target: list[list[int]], max_distance: int | None,
) -> list[int]:
    """Share the exact pre-edit distance across same-state candidates."""
    distance = boundary_token_edit_distance(before, target, max_distance)
    return [
        distance - boundary_token_edit_distance(
            outcome, target,
            None if max_distance is None else max_distance + 1,
        )
        for outcome in outcomes
    ]


__all__ = [
    "boundary_token_edit_distance", "boundary_token_sequence",
    "exact_one_step_advantage", "exact_one_step_advantages",
    "token_levenshtein",
]
