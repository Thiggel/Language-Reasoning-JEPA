"""Symbolic environment over an iGSM problem.

The environment is the ground-truth "world": actions resolve variables,
steps render outcome sentences, and success is checked symbolically. The
planner may query the action *interface* (feasible actions and their
intent phrases) but never the consequences — those it must predict in
latent space.
"""

from __future__ import annotations

from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import action_phrase, step_sentence


INVALID_ACTION_OUTCOME = (
    "The proposed action is invalid and the state is unchanged ."
)
INVALID_ACTION_FAILURE = (
    "The proposed action is invalid and the problem can no longer be solved ."
)


class SymbolicEnv:
    def __init__(self, problem: Problem, invalid_action_mode: str = "noop"):
        if invalid_action_mode not in {"noop", "failure"}:
            raise ValueError(f"unknown invalid action mode: {invalid_action_mode}")
        self.p = problem
        self.resolved: list[int] = []
        self.invalid_action_mode = invalid_action_mode
        self.failed = False

    @property
    def resolved_set(self) -> set[int]:
        return set(self.resolved)

    def clone(self) -> "SymbolicEnv":
        c = SymbolicEnv(self.p, self.invalid_action_mode)
        c.resolved = list(self.resolved)
        c.failed = self.failed
        return c

    def feasible_actions(self) -> list[int]:
        """Unresolved variables whose parents are all resolved."""
        if self.failed:
            return []
        done = self.resolved_set
        return [
            v.idx
            for v in self.p.vars
            if v.idx not in done and all(pa in done for pa in v.parents)
        ]

    def action_text(self, idx: int) -> str:
        return action_phrase(self.p, idx)

    def step(self, idx: int) -> str:
        if idx not in self.feasible_actions():
            raise ValueError(f"infeasible action {idx}")
        self.resolved.append(idx)
        return step_sentence(self.p, idx)

    def step_or_invalid(self, idx: int) -> str:
        """Execute an action from the full catalogue without exposing a menu."""
        if self.failed:
            return INVALID_ACTION_FAILURE
        if idx not in self.feasible_actions():
            if self.invalid_action_mode == "failure":
                self.failed = True
                return INVALID_ACTION_FAILURE
            return INVALID_ACTION_OUTCOME
        return self.step(idx)

    @property
    def solved(self) -> bool:
        return self.p.query in self.resolved_set

    def remaining_necessary(self) -> int:
        """Number of still-unresolved ancestors of the query (query incl.)."""
        return len(self.p.query_ancestors - self.resolved_set)
