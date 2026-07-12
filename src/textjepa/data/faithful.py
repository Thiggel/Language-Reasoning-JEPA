"""100%-faithful iGSM: a thin adapter over the OFFICIAL generator
(facebookresearch/iGSM, MIT, vendored in third_party/iGSM).

Problems, prompt text, question, solution steps and answers all come from
the reference implementation (IdGen). Our additions are interface-only:
- FaithfulEnv: the planning interface (feasible parameters, intent
  phrases, step rendering in the reference solution grammar);
- FaithfulDataset: our batch schema over their traces.

Fidelity is machine-checked: env-generated full solutions must pass the
official checker ``tools.tools_test.true_correct``
(scripts/validate_faithful.py, also run in tests).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from textjepa.data.vocab import EDIT_WORDS, Vocab

_IGSM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "iGSM"
if str(_IGSM_ROOT) not in sys.path:
    sys.path.insert(0, str(_IGSM_ROOT))

OP_LABELS = {"const": 0, "sum": 1, "diff": 2, "mul": 3}


def _fix_seed(key: str) -> None:
    import hashlib

    from tools.tools import fix_seed

    # stable across processes (builtin hash() is salted per process)
    h = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "little")
    fix_seed(h % (2**31 - 1))


def gen_problem(key: str, max_op: int, max_edge: int, op_range=(None, None)):
    """Deterministic official-generator call; optional rejection on n_op."""
    from data_gen.pretrain.id_gen import IdGen

    _fix_seed(key)
    for attempt in range(50):
        g = IdGen(max_op=max_op, max_edge=max_edge, perm_level=5,
                  detail_level=0)
        g.gen_prob(list(range(23)), p_format="pq")
        lo, hi = op_range
        n = g.problem.n_op
        if (lo is None or n >= lo) and (hi is None or n <= hi):
            return g
    return g


class FaithfulProblem:
    def __init__(self, gen):
        import networkx as nx
        from tools.tools import tokenizer

        p = gen.problem
        self.p = p
        self.text = tokenizer.decode(gen.prob_token).strip()
        self.sol_text = tokenizer.decode(gen.sol_token).strip()
        self.answer = int(p.ans)
        # exclude the RNG pseudo-node (l = -1): it feeds random
        # constants into expressions and is never itself "defined"
        self.params = [q for q in p.template.nodes if q[0] != -1]
        self.deps = {
            q: [d for d in p.template.predecessors(q) if d[0] != -1]
            for q in self.params
        }
        self.query = p.ques_idx
        self.necessary = (
            {q for q in nx.ancestors(p.template, self.query) if q[0] != -1}
            | {self.query}
        )
        # reference solutions name parameters WITHOUT the 'each' prefix
        self.names = {
            q: p.get_param(q).replace("each ", "") for q in self.params
        }
        self.values = {q: int(p.sketch[q].get_value.a) for q in self.params
                       if q in p.sketch}
        # prompt sentences: the reference text is sentence-per-parameter,
        # final sentence is the question
        parts = [s.strip() + "." for s in self.text.split(". ") if s.strip()]
        parts[-1] = parts[-1].rstrip(".")
        self.prompt_sentences = parts

    def op_label(self, q) -> int:
        exp = self.p.sketch[q]
        if not exp.param_list:
            return OP_LABELS["const"]
        return OP_LABELS.get(exp.op, 1)


class FaithfulEnv:
    """Planning interface over the official dependency graph. Step
    sentences are rendered by the REFERENCE renderer (Problem.to_sol)
    on a state-reset copy of the problem — grammar fidelity by
    construction."""

    def __init__(self, problem: FaithfulProblem, _p2=None):
        import copy

        self.fp = problem
        self.resolved: list = []
        if _p2 is not None:
            self.p2 = _p2
        else:
            p2 = copy.deepcopy(problem.p)
            p2.solution = []
            p2.name_dict = {}
            p2.lookup = {
                k: v for k, v in p2.lookup.items() if isinstance(k, tuple)
            }
            from math_gen.problem_gen import feasible_symbols

            p2.symbols = copy.deepcopy(feasible_symbols)
            self.p2 = p2

    @property
    def resolved_set(self) -> set:
        return set(self.resolved)

    def clone(self) -> "FaithfulEnv":
        import copy

        c = FaithfulEnv(self.fp, _p2=copy.deepcopy(self.p2))
        c.resolved = list(self.resolved)
        return c

    def feasible_actions(self) -> list:
        done = self.resolved_set
        return [q for q in self.fp.params
                if q not in done
                and all(d in done for d in self.fp.deps[q])
                and q in self.fp.p.sketch]

    def action_text(self, q) -> str:
        return f"Define {self.fp.names[q]} ."

    def step(self, q) -> str:
        assert q in self.feasible_actions(), f"infeasible {q}"
        self.p2.to_sol(self.p2.sketch[q], append=True)
        self.resolved.append(q)
        return self.p2.solution[-1] + "."

    @property
    def solved(self) -> bool:
        return self.fp.query in self.resolved_set

    def remaining_necessary(self) -> int:
        return len(self.fp.necessary - self.resolved_set)


class FaithfulDataset(Dataset):
    """Batch schema compatible with the discourse collate; traces follow
    the official minimal solution order with optional distractor detours."""

    def __init__(
        self,
        vocab: Vocab,
        size: int,
        seed: int,
        max_op: int = 15,
        max_edge: int = 20,
        op_range: tuple = (3, 15),
        distractor_prob: float = 0.15,
        max_distractors: int = 2,
        n_alt: int = 0,
        **_,
    ):
        self.n_alt = n_alt
        self.vocab = vocab
        self.size = size
        self.seed = seed
        self.max_op = max_op
        self.max_edge = max_edge
        self.op_range = tuple(op_range)
        self.distractor_prob = distractor_prob
        self.max_distractors = max_distractors

    def __len__(self) -> int:
        return self.size

    def problem(self, index: int):
        gen = gen_problem(
            f"{self.seed}:{index}", self.max_op, self.max_edge, self.op_range
        )
        return FaithfulProblem(gen), random.Random(f"{self.seed}:{index}:t")

    def __getitem__(self, index: int) -> dict:
        fp, rng = self.problem(index)
        env = FaithfulEnv(fp)
        steps, actions, op, value, remaining, resolved_n, necessary = (
            [], [], [], [], [], [], []
        )
        n_distr = 0
        pidx = {q: i for i, q in enumerate(fp.params)}
        var_idx = []
        alt_actions: list = []
        alt_remaining: list = []
        while not env.solved:
            feas = env.feasible_actions()
            nec = [q for q in feas if q in fp.necessary]
            distr = [q for q in feas if q not in fp.necessary]
            use_d = (distr and n_distr < self.max_distractors
                     and rng.random() < self.distractor_prob)
            q = rng.choice(distr) if use_d else rng.choice(nec)
            n_distr += int(q not in fp.necessary)
            if self.n_alt:
                done = env.resolved_set
                others = [a for a in feas if a != q]
                rng.shuffle(others)
                alts = others[: self.n_alt]
                alt_actions.append(
                    [self.vocab.encode(env.action_text(a)) for a in alts]
                )
                alt_remaining.append(
                    [len(fp.necessary - (done | {a})) for a in alts]
                )
            actions.append(self.vocab.encode(env.action_text(q)))
            steps.append(self.vocab.encode(env.step(q)))
            op.append(fp.op_label(q))
            value.append(fp.values[q])
            remaining.append(env.remaining_necessary())
            resolved_n.append(len(env.resolved))
            necessary.append(int(q in fp.necessary))
            var_idx.append(min(pidx[q], 11))
        prompt = [self.vocab.encode(s) for s in fp.prompt_sentences]
        out = {
            "prompt": prompt, "steps": steps, "actions": actions,
            "op": op, "value": value, "remaining": remaining,
            "resolved_n": resolved_n, "necessary": necessary,
            "answer": fp.answer, "n_necessary": len(fp.necessary),
            "n_vars": len(fp.params), "index": index,
            "var_idx": var_idx, "query_idx": min(pidx[fp.query], 11),
            "ancestors": sorted(
                min(pidx[q], 11) for q in fp.necessary
            ),
        }
        if self.n_alt:
            out["alt_actions"] = alt_actions
            out["alt_remaining"] = alt_remaining
        return out


def build_faithful_vocab(n_scan: int = 1500, max_op: int = 21,
                         max_edge: int = 28) -> Vocab:
    """Vocabulary from a deterministic scan of the official generator's
    output space (all worlds' names appear quickly) + solution symbols."""
    words: set[str] = set(EDIT_WORDS)
    for i in range(n_scan):
        gen = gen_problem(f"vocab:{i}", max_op, max_edge)
        fp = FaithfulProblem(gen)
        for s in fp.prompt_sentences:
            words.update(s.split())
        words.update(fp.sol_text.split())
    from math_gen.problem_gen import Problem  # noqa

    words.update(str(i) for i in range(23))
    words.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    words.update(["Define", "as", "so", "=", "+", "-", "*", ";", ".", "?"])
    return Vocab(sorted(words))


_VOCAB_CACHE = Path(__file__).resolve().parents[3] / "configs" / "faithful_vocab.txt"


def cached_faithful_vocab() -> Vocab:
    if _VOCAB_CACHE.exists():
        return Vocab(_VOCAB_CACHE.read_text().split("\n"))
    v = build_faithful_vocab()
    _VOCAB_CACHE.write_text(
        "\n".join(t for t in v.token_to_id if not t.startswith("<"))
    )
    return v
