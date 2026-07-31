"""Conservative symbolic verification for rendered iGSM reasoning steps."""

from __future__ import annotations

from dataclasses import dataclass
import re


_OPERATION = re.compile(
    r"(?:so\s+)?(?:the\s+number\s+of\s+|computing\s+)?"
    r"(?P<target>[a-z][a-z ]*?)\s*(?:is|:)\s*"
    r"(?P<left>-?\d+)\s*"
    r"(?P<op>plus|minus|times)\s*"
    r"(?P<right>-?\d+)\s*(?:=|gives)\s*"
    r"(?P<result>-?\d+)",
    re.IGNORECASE,
)
_FINAL = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedOperation:
    target: str
    left: int
    operation: str
    right: int
    result: int


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().strip().split())


def parse_rendered_operation(text: str) -> ParsedOperation | None:
    match = _OPERATION.search(text)
    if match is None:
        return None
    return ParsedOperation(
        target=_normalize_name(match.group("target")),
        left=int(match.group("left")),
        operation=match.group("op").lower(),
        right=int(match.group("right")),
        result=int(match.group("result")),
    )


def operation_matches_expected(
    candidate: str, expected: str, *, modulus: int = 23
) -> bool:
    """Check a generated operation against the verified environment action."""

    proposed = parse_rendered_operation(candidate)
    reference = parse_rendered_operation(expected)
    if proposed is None or reference is None or proposed.target != reference.target:
        return False
    if proposed.operation != reference.operation:
        return False
    proposed_operands = (proposed.left % modulus, proposed.right % modulus)
    reference_operands = (reference.left % modulus, reference.right % modulus)
    if proposed.operation in {"plus", "times"}:
        if sorted(proposed_operands) != sorted(reference_operands):
            return False
    elif proposed_operands != reference_operands:
        return False
    if proposed.operation == "plus":
        computed = (proposed.left + proposed.right) % modulus
    elif proposed.operation == "minus":
        computed = (proposed.left - proposed.right) % modulus
    else:
        computed = (proposed.left * proposed.right) % modulus
    return (
        proposed.result % modulus == computed
        and proposed.result % modulus == reference.result % modulus
    )


def final_answer_matches(candidate: str, answer: int, *, modulus: int = 23) -> bool:
    match = _FINAL.search(candidate)
    return match is not None and int(match.group(1)) % modulus == answer % modulus


def achieved_next_state_id(
    candidate: str,
    record: dict,
    boundary_index: int,
) -> int:
    """Return the verified successor ID or ``-1`` for unsupported text."""

    operations = record["reasoning_operations"]
    states = record["canonical_state_ids"]
    if boundary_index < 0 or boundary_index >= len(states) - 1:
        raise ValueError("boundary index lies outside the symbolic trace")
    if boundary_index < len(operations):
        valid = operation_matches_expected(
            candidate, operations[boundary_index]
        )
    else:
        valid = final_answer_matches(candidate, int(record["answer"]))
    return int(states[boundary_index + 1]) if valid else -1
