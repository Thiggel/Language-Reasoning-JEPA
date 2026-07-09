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


class SymbolicEnv:
    def __init__(self, problem: Problem):
        self.p = problem
        self.resolved: list[int] = []

    @property
    def resolved_set(self) -> set[int]:
        return set(self.resolved)

    def clone(self) -> "SymbolicEnv":
        c = SymbolicEnv(self.p)
        c.resolved = list(self.resolved)
        return c

    def feasible_actions(self) -> list[int]:
        """Unresolved variables whose parents are all resolved."""
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

    @property
    def solved(self) -> bool:
        return self.p.query in self.resolved_set

    def remaining_necessary(self) -> int:
        """Number of still-unresolved ancestors of the query (query incl.)."""
        return len(self.p.query_ancestors - self.resolved_set)
