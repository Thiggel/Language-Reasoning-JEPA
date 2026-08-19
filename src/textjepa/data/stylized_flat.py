"""Stylized-iGSM adapter for the flat-backbone intent JEPA.

The flat stack (``data/flat_stream.py``, ``planning/flat_search.py``,
``scripts/plan_flat.py``, ``scripts/probe_flat_energy_auc.py``) was written
against the faithful generator's problem/environment surface
(``fp.prompt_sentences``, ``fp.action_order``, ``fp.necessary``,
``fp.make_env()``).  The stylized generator exposes the same information in a
different shape, so this module presents it through that one surface --
nothing about the flat model, losses, planner, or probes changes between the
two data settings.

Also provides the stylized vocabulary used by the flat stack: the historical
``igsm.dataset.build_vocab`` list plus the words of the "invalid action"
outcome sentence, which menu-free training and evaluation render.  It is a
separate builder so existing stylized checkpoints keep their token ids.
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab
from textjepa.data.igsm.env import INVALID_ACTION_FAILURE, INVALID_ACTION_OUTCOME, SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.data.vocab import Vocab

INVALID_ACTION_WORDS = (
    INVALID_ACTION_OUTCOME.split() + INVALID_ACTION_FAILURE.split()
)


def build_flat_stylized_vocab(modulus: int = 23, adjectives=None, nouns=None) -> Vocab:
    v = build_vocab(modulus, adjectives, nouns)
    tokens = [t for t in v.token_to_id if not t.startswith("<")]
    extra = [w for w in INVALID_ACTION_WORDS if w not in set(tokens)]
    return Vocab(tokens + sorted(set(extra)))


class StylizedFlatEnv:
    """``FaithfulEnv``-shaped view of :class:`SymbolicEnv`."""

    def __init__(self, problem: "StylizedFlatProblem", env: SymbolicEnv | None = None):
        self.fp = problem
        self.env = env if env is not None else SymbolicEnv(problem.p)

    @property
    def resolved(self) -> list:
        return self.env.resolved

    @property
    def resolved_set(self) -> set:
        return self.env.resolved_set

    def clone(self) -> "StylizedFlatEnv":
        return StylizedFlatEnv(self.fp, self.env.clone())

    def feasible_actions(self) -> list:
        return self.env.feasible_actions()

    def action_text(self, q) -> str:
        return self.env.action_text(q)

    def step(self, q) -> str:
        return self.env.step(q)

    def step_or_invalid(self, q) -> str:
        return self.env.step_or_invalid(q)

    @property
    def solved(self) -> bool:
        return self.env.solved

    def remaining_necessary(self) -> int:
        return self.env.remaining_necessary()


class StylizedFlatProblem:
    """One stylized problem with the faithful problem's attribute names.

    ``prompt_sentences`` is drawn from a dedicated RNG stream (the training
    dataset shuffles definition order with the item RNG; only the order
    differs, so eval prompts are distributionally identical).
    """

    def __init__(self, p, key: str = ""):
        self.p = p
        self.prompt_sentences = prompt_sentences(
            p, random.Random(f"stylized-flat-prompt:{key}")
        )
        self.params = [v.idx for v in p.vars]
        self.action_order = list(self.params)
        self.necessary = set(p.query_ancestors)
        self.query = p.query
        self.answer = p.answer
        self.values = p.values

    def make_env(self) -> StylizedFlatEnv:
        return StylizedFlatEnv(self)


class StylizedFlatDataset(Dataset):
    """``IGSMDataset`` items plus the flat problem/env surface."""

    INVALID_OUTCOME = INVALID_ACTION_OUTCOME

    def __init__(self, base: IGSMDataset):
        self.base = base
        self.vocab = base.vocab
        self.all_action_supervision = base.all_action_supervision

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        return self.base[index]

    def problem(self, index: int):
        p, rng = self.base.problem(index)
        return StylizedFlatProblem(p, key=f"{self.base.seed}:{index}"), rng
