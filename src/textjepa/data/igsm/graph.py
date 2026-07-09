"""iGSM-style synthetic reasoning problems.

A problem is a DAG of named integer quantities. Leaves carry constants;
internal variables combine two parents with modular arithmetic. The query
asks for one variable's value; solving it requires resolving exactly its
ancestor set, while the remaining variables act as distractors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cached_property

OPS = ("add", "sub", "mul")
OP_WORDS = {"add": "plus", "sub": "minus", "mul": "times"}
CONST_OP = "const"


@dataclass(frozen=True)
class Var:
    idx: int
    adj: str
    noun: str
    op: str  # CONST_OP or one of OPS
    parents: tuple[int, ...] = ()
    const: int = 0

    @property
    def name(self) -> str:
        return f"{self.adj} {self.noun}"

    @property
    def is_leaf(self) -> bool:
        return self.op == CONST_OP


@dataclass(frozen=True)
class Problem:
    vars: tuple[Var, ...]
    query: int
    modulus: int

    @cached_property
    def values(self) -> tuple[int, ...]:
        vals: list[int | None] = [None] * len(self.vars)

        def ev(i: int) -> int:
            if vals[i] is None:
                v = self.vars[i]
                if v.is_leaf:
                    vals[i] = v.const % self.modulus
                else:
                    a, b = (ev(p) for p in v.parents)
                    if v.op == "add":
                        vals[i] = (a + b) % self.modulus
                    elif v.op == "sub":
                        vals[i] = (a - b) % self.modulus
                    else:
                        vals[i] = (a * b) % self.modulus
            return vals[i]

        return tuple(ev(i) for i in range(len(self.vars)))

    @cached_property
    def query_ancestors(self) -> frozenset[int]:
        """Ancestors of the query, query included: the necessary computation."""
        out: set[int] = set()
        stack = [self.query]
        while stack:
            i = stack.pop()
            if i not in out:
                out.add(i)
                stack.extend(self.vars[i].parents)
        return frozenset(out)

    @property
    def answer(self) -> int:
        return self.values[self.query]

    @property
    def n_necessary_steps(self) -> int:
        return len(self.query_ancestors)


def sample_problem(
    rng: random.Random,
    adjectives: list[str],
    nouns: list[str],
    modulus: int = 23,
    n_vars_range: tuple[int, int] = (6, 12),
    leaf_prob: float = 0.35,
    steps_range: tuple[int, int] = (3, 9),
    max_tries: int = 50,
) -> Problem:
    """Rejection-sample a problem whose necessary trace length is in range."""
    for _ in range(max_tries):
        p = _sample_once(rng, adjectives, nouns, modulus, n_vars_range, leaf_prob)
        if steps_range[0] <= p.n_necessary_steps <= steps_range[1]:
            return p
    return p  # rare fallback: last sample regardless of length


def _sample_once(
    rng: random.Random,
    adjectives: list[str],
    nouns: list[str],
    modulus: int,
    n_vars_range: tuple[int, int],
    leaf_prob: float,
) -> Problem:
    n = rng.randint(*n_vars_range)
    names = rng.sample([(a, b) for a in adjectives for b in nouns], n)
    vars_: list[Var] = []
    for i, (adj, noun) in enumerate(names):
        if i < 2 or rng.random() < leaf_prob:
            vars_.append(Var(i, adj, noun, CONST_OP, const=rng.randrange(modulus)))
        else:
            pa, pb = rng.sample(range(i), 2)
            vars_.append(Var(i, adj, noun, rng.choice(OPS), parents=(pa, pb)))
    internal = [v.idx for v in vars_ if not v.is_leaf]
    query = rng.choice(internal) if internal else n - 1
    return Problem(tuple(vars_), query, modulus)
