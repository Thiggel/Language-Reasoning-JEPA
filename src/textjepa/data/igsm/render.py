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


def answer_sentence(p: Problem) -> str:
    return f"the answer is {p.answer} ."


TEMPLATE_WORDS = [
    "the", "number", "of", "is", "equals", "plus", "minus", "times", ".",
    "how", "many", "are", "there", "?", "so", "=", "look", "up", "derive",
    "from", "and", "answer",
]
