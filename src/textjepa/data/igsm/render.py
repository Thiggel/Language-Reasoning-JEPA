"""Natural-language rendering of iGSM problems, steps, and action phrases.

Reasoning *steps* state outcomes (they include computed values); *action
phrases* state intent only, so a planner can encode candidate actions
without knowing their consequences.
"""

from __future__ import annotations

import random

from textjepa.data.igsm.graph import OP_WORDS, Problem


def definition_sentence(p: Problem, idx: int) -> str:
    v = p.vars[idx]
    if v.is_leaf:
        return f"the number of {v.name} is {v.const} ."
    a, b = (p.vars[j] for j in v.parents)
    return (
        f"the number of {v.name} equals the number of {a.name} "
        f"{OP_WORDS[v.op]} the number of {b.name} ."
    )


def question_sentence(p: Problem) -> str:
    return f"how many {p.vars[p.query].name} are there ?"


def prompt_sentences(p: Problem, rng: random.Random) -> list[str]:
    """All definitions in random order, question last."""
    order = list(range(len(p.vars)))
    rng.shuffle(order)
    return [definition_sentence(p, i) for i in order] + [question_sentence(p)]


def step_sentence(p: Problem, idx: int) -> str:
    """Outcome sentence for resolving variable ``idx``."""
    v = p.vars[idx]
    if v.is_leaf:
        return f"so the number of {v.name} is {v.const % p.modulus} ."
    a, b = v.parents
    return (
        f"so the number of {v.name} is {p.values[a]} {OP_WORDS[v.op]} "
        f"{p.values[b]} = {p.values[idx]} ."
    )


def action_phrase(p: Problem, idx: int) -> str:
    """Intent sentence for resolving variable ``idx`` (no outcome leaked)."""
    v = p.vars[idx]
    if v.is_leaf:
        return f"look up the number of {v.name} ."
    a, b = (p.vars[j] for j in v.parents)
    return (
        f"derive {v.name} from {a.name} {OP_WORDS[v.op]} {b.name} ."
    )


def catalogue_phrases(p: Problem) -> list[str]:
    """Every action phrase of a problem, indexed by variable."""
    return [action_phrase(p, v.idx) for v in p.vars]


WORD_OPS = {word: op for op, word in OP_WORDS.items()}
_LOOKUP_PREFIX = ["look", "up", "the", "number", "of"]


def parse_action_phrase(p: Problem, text: str) -> int | None:
    """Inverse of :func:`action_phrase`: ground a phrase in ``p``'s actions.

    Returns the action index whose intent phrase the text denotes, or ``None``
    if the text is not a well-formed intent phrase for this problem (unknown
    variable name, wrong arity, or an operation/parent pair that contradicts
    the stated definition).  Used by open-ended (generator) planning, where
    proposals are free token sequences that must be parsed or discarded.
    """
    words = text.split()
    while words and words[-1] == ".":
        words = words[:-1]
    names = {v.name: v.idx for v in p.vars}
    if words[: len(_LOOKUP_PREFIX)] == _LOOKUP_PREFIX:
        idx = names.get(" ".join(words[len(_LOOKUP_PREFIX):]))
        if idx is None or not p.vars[idx].is_leaf:
            return None
        return idx
    if not words or words[0] != "derive" or "from" not in words:
        return None
    split = words.index("from")
    idx = names.get(" ".join(words[1:split]))
    rest = words[split + 1:]
    positions = [i for i, word in enumerate(rest) if word in WORD_OPS]
    if idx is None or len(positions) != 1:
        return None
    at = positions[0]
    left = names.get(" ".join(rest[:at]))
    right = names.get(" ".join(rest[at + 1:]))
    v = p.vars[idx]
    if left is None or right is None or v.is_leaf:
        return None
    if v.op != WORD_OPS[rest[at]] or v.parents != (left, right):
        return None
    return idx


def answer_sentence(p: Problem) -> str:
    return f"the answer is {p.answer} ."


TEMPLATE_WORDS = [
    "the", "number", "of", "is", "equals", "plus", "minus", "times", ".",
    "how", "many", "are", "there", "?", "so", "=", "look", "up", "derive",
    "from", "and", "answer",
]
