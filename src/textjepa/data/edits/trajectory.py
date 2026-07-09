"""Text-buffer world: corrupted solution drafts repaired by span edits.

The buffer state is a draft solution to an iGSM problem. Corruptions have
*propagated consequences*: corrupting one step's value rewrites every
downstream step consistently (change an assumption -> later derivations
change), so a single root cause manifests as several defects. Edits are
DELETE / INSERT / REPLACE with intent phrases that never leak outcomes.

A step is defective iff its sentence differs from the true step sentence;
missing necessary steps and extra distractor steps are also defects.
Solved = zero defects (buffer == perfect draft, order-insensitive).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from textjepa.data.igsm.graph import OP_WORDS, Problem
from textjepa.data.igsm.render import step_sentence

EDIT_OPS = ("delete", "insert", "replace")
EDIT_OP_LABELS = {op: i for i, op in enumerate(EDIT_OPS)}


@dataclass(frozen=True)
class Edit:
    kind: str  # one of EDIT_OPS
    pos: int  # buffer index (delete/replace) or insertion index (insert)
    var: int | None = None


@dataclass
class BufferStep:
    var: int
    text: str


def topo_necessary(p: Problem) -> list[int]:
    """Canonical topological order of the necessary computation."""
    order, resolved = [], set()
    todo = set(p.query_ancestors)
    while todo:
        ready = sorted(
            i for i in todo if all(pa in resolved for pa in p.vars[i].parents)
        )
        order.extend(ready)
        resolved.update(ready)
        todo -= set(ready)
    return order


def corrupted_values(p: Problem, corrupted: dict[int, int]) -> list[int]:
    """Variable values where corruptions propagate through descendants."""
    vals: list[int | None] = [None] * len(p.vars)

    def ev(i: int) -> int:
        if vals[i] is None:
            if i in corrupted:
                vals[i] = corrupted[i]
            elif p.vars[i].is_leaf:
                vals[i] = p.vars[i].const % p.modulus
            else:
                a, b = (ev(x) for x in p.vars[i].parents)
                op = p.vars[i].op
                r = a + b if op == "add" else a - b if op == "sub" else a * b
                vals[i] = r % p.modulus
        return vals[i]

    return [ev(i) for i in range(len(p.vars))]


def stated_sentence(p: Problem, idx: int, vals: list[int]) -> str:
    v = p.vars[idx]
    if v.is_leaf:
        return f"so the number of {v.name} is {vals[idx]} ."
    a, b = v.parents
    return (
        f"so the number of {v.name} is {vals[a]} {OP_WORDS[v.op]} "
        f"{vals[b]} = {vals[idx]} ."
    )


class EditEnv:
    def __init__(
        self,
        problem: Problem,
        rng: random.Random,
        max_wrong: int = 2,
        max_missing: int = 1,
        max_extra: int = 1,
        min_defects: int = 1,
    ):
        self.p = problem
        necessary = topo_necessary(problem)

        internal = [i for i in necessary if not problem.vars[i].is_leaf]
        n_wrong = rng.randint(0, min(max_wrong, len(internal)))
        corrupted = {}
        for i in rng.sample(internal, n_wrong):
            true = problem.values[i]
            corrupted[i] = rng.choice([v for v in range(problem.modulus) if v != true])
        vals = corrupted_values(problem, corrupted)

        removable = [i for i in necessary if i != problem.query]
        n_missing = rng.randint(0, min(max_missing, len(removable)))
        missing = set(rng.sample(removable, n_missing))

        distractors = [v.idx for v in problem.vars if v.idx not in problem.query_ancestors]
        n_extra = rng.randint(0, min(max_extra, len(distractors)))
        extras = rng.sample(distractors, n_extra)

        self.buffer: list[BufferStep] = [
            BufferStep(i, stated_sentence(problem, i, vals))
            for i in necessary
            if i not in missing
        ]
        for e in extras:
            self.buffer.insert(
                rng.randint(0, len(self.buffer)), BufferStep(e, step_sentence(problem, e))
            )
        if self.n_defects() < min_defects and internal:
            i = rng.choice(internal)
            wrong = (problem.values[i] + 1 + rng.randrange(problem.modulus - 1)) % problem.modulus
            vals2 = corrupted_values(problem, {i: wrong})
            for b in self.buffer:
                if b.var in problem.query_ancestors:
                    b.text = stated_sentence(problem, b.var, vals2)

    # ------------------------------------------------------------------ #
    def _true_text(self, var: int) -> str:
        return step_sentence(self.p, var)

    def is_defect(self, b: BufferStep) -> bool:
        return b.var not in self.p.query_ancestors or b.text != self._true_text(b.var)

    def missing_necessary(self) -> list[int]:
        present = {b.var for b in self.buffer}
        return [i for i in topo_necessary(self.p) if i not in present]

    def n_defects(self) -> int:
        return sum(self.is_defect(b) for b in self.buffer) + len(self.missing_necessary())

    @property
    def solved(self) -> bool:
        return self.n_defects() == 0

    def sentences(self) -> list[str]:
        return [b.text for b in self.buffer]

    # ------------------------------------------------------------------ #
    def insert_index(self, var: int) -> int:
        """Topologically sound insertion point for a missing step."""
        last = -1
        ancestors = {
            a for a in self.p.query_ancestors if a != var
        } & self._strict_ancestors(var)
        for k, b in enumerate(self.buffer):
            if b.var in ancestors:
                last = k
        return last + 1

    def _strict_ancestors(self, var: int) -> set[int]:
        out: set[int] = set()
        stack = list(self.p.vars[var].parents)
        while stack:
            i = stack.pop()
            if i not in out:
                out.add(i)
                stack.extend(self.p.vars[i].parents)
        return out

    def candidate_edits(self, include_harmful: bool = True, rng=None) -> list[Edit]:
        cands = []
        for pos, b in enumerate(self.buffer):
            cands.append(Edit("delete", pos))
            cands.append(Edit("replace", pos, b.var))
        for var in self.missing_necessary():
            cands.append(Edit("insert", self.insert_index(var), var))
        if include_harmful:
            present = {b.var for b in self.buffer}
            harmful = [
                v.idx
                for v in self.p.vars
                if v.idx not in self.p.query_ancestors and v.idx not in present
            ]
            if rng is not None:
                harmful = rng.sample(harmful, min(2, len(harmful)))
            cands += [
                Edit("insert", len(self.buffer), var) for var in harmful[:2]
            ]
        return cands

    def fixing_edits(self) -> list[Edit]:
        fixes = [
            Edit("delete", pos)
            if b.var not in self.p.query_ancestors
            else Edit("replace", pos, b.var)
            for pos, b in enumerate(self.buffer)
            if self.is_defect(b)
        ]
        fixes += [
            Edit("insert", self.insert_index(v), v) for v in self.missing_necessary()
        ]
        return fixes

    def intent_text(self, e: Edit) -> str:
        if e.kind == "delete":
            return f"delete step {e.pos + 1} ."
        if e.kind == "replace":
            return f"replace step {e.pos + 1} : recompute {self.p.vars[e.var].name} ."
        name = self.p.vars[e.var].name
        if e.pos == 0:
            return f"insert {name} at the start ."
        return f"insert {name} after step {e.pos} ."

    def apply(self, e: Edit) -> None:
        if e.kind == "delete":
            self.buffer.pop(e.pos)
        elif e.kind == "replace":
            self.buffer[e.pos] = BufferStep(e.var, self._true_text(e.var))
        else:
            self.buffer.insert(e.pos, BufferStep(e.var, self._true_text(e.var)))

    def stated_query_value(self) -> int:
        """Value the buffer currently claims for the query (modulus = absent)."""
        for b in reversed(self.buffer):
            if b.var == self.p.query:
                tokens = b.text.split()
                return int(tokens[-2])
        return self.p.modulus
