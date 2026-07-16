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
        # The reference graph iterates parameters in a near-topological order.
        # Exposing that order as the action menu makes "take the first
        # feasible action" an accidental solution policy.  Preserve the
        # official problem itself but present every model with one stable,
        # problem-specific shuffled action order.
        self.action_order = list(self.params)
        random.Random(f"faithful-action-menu:{self.text}").shuffle(
            self.action_order
        )
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
        return [q for q in self.fp.action_order
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


def enumerate_faithful_action_sequences(
    env: FaithfulEnv, depth: int, cap: int = 256
) -> list[list]:
    """Enumerate fixed-depth feasible chunks from a faithful environment."""
    frontier: list[tuple[list, FaithfulEnv]] = [([], env.clone())]
    for _ in range(depth):
        nxt = []
        for sequence, current in frontier:
            for action in current.feasible_actions():
                clone = current.clone()
                clone.step(action)
                nxt.append((sequence + [action], clone))
        frontier = nxt[:cap]
        if not frontier:
            break
    return [sequence for sequence, _ in frontier if len(sequence) == depth]


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
        geo_rank_k: int = 0,
        geo_rank_horizon: int = 1,
        geo_rank_rollouts: int = 1,
        geo_rank_policy: str = "random",
        geo_rank_beam_width: int = 1,
        macro_alt_k: int = 0,
        macro_alt_horizon: int = 3,
        all_action_supervision: bool = False,
        **_,
    ):
        self.n_alt = n_alt
        self.geo_rank_k = geo_rank_k
        self.geo_rank_horizon = max(1, int(geo_rank_horizon))
        self.geo_rank_rollouts = max(1, int(geo_rank_rollouts))
        self.geo_rank_policy = str(geo_rank_policy)
        self.geo_rank_beam_width = max(1, int(geo_rank_beam_width))
        self.macro_alt_k = max(0, int(macro_alt_k))
        self.macro_alt_horizon = max(1, int(macro_alt_horizon))
        self.all_action_supervision = bool(all_action_supervision)
        if self.geo_rank_policy not in {"random", "greedy"}:
            raise ValueError(f"unknown geo_rank_policy: {self.geo_rank_policy}")
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
        # Alternative outcomes are an optional supervision view, not part of
        # trajectory generation.  Isolate their randomness so n_alt=0 and
        # n_alt>0 remain exactly paired on every on-trajectory field.
        alt_rng = random.Random(f"{self.seed}:{index}:alt")
        env = FaithfulEnv(fp)
        steps, actions, op, value, remaining, resolved_n, necessary = (
            [], [], [], [], [], [], []
        )
        n_distr = 0
        pidx = {q: i for i, q in enumerate(fp.params)}
        var_idx = []
        alt_actions: list = []
        alt_steps: list = []
        alt_remaining: list = []
        action_feasible: list[list[int]] = []
        trace: list = []
        while not env.solved:
            feas = env.feasible_actions()
            if self.all_action_supervision:
                feasible_set = set(feas)
                action_feasible.append([
                    int(candidate in feasible_set)
                    for candidate in fp.action_order
                ])
            nec = [q for q in feas if q in fp.necessary]
            distr = [q for q in feas if q not in fp.necessary]
            use_d = (distr and n_distr < self.max_distractors
                     and rng.random() < self.distractor_prob)
            q = rng.choice(distr) if use_d else rng.choice(nec)
            trace.append(q)
            n_distr += int(q not in fp.necessary)
            if self.n_alt:
                import numpy as np

                done = env.resolved_set
                others = [a for a in feas if a != q]
                alt_rng.shuffle(others)
                alts = others[: self.n_alt]
                alt_actions.append(
                    [self.vocab.encode(env.action_text(a)) for a in alts]
                )
                # Problem.to_sol draws temporary variable names from the
                # process-global Python RNG.  Counterfactual rendering must
                # not perturb the factual renderer (or the next action).
                py_state = random.getstate()
                np_state = np.random.get_state()
                try:
                    rendered_alts = [
                        self.vocab.encode(env.clone().step(a)) for a in alts
                    ]
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                alt_steps.append(rendered_alts)
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
        ga = {}
        if self.geo_rank_k and len(trace) > 1:
            t_star = rng.randrange(len(trace))
            env2 = FaithfulEnv(fp)
            for q in trace[:t_star]:
                env2.step(q)
            executed = trace[t_star]
            alternatives = [q for q in env2.feasible_actions() if q != executed]
            rng.shuffle(alternatives)
            alternatives = alternatives[: self.geo_rank_k]
            if alternatives:
                candidates = [executed, *alternatives]
                ga = {
                    "ga_t": t_star,
                    "ga_horizon": self.geo_rank_horizon,
                    "ga_beam_width": self.geo_rank_beam_width,
                    "ga_candidate_objects": candidates,
                    "ga_alt_actions": [
                        self.vocab.encode(env2.action_text(q))
                        for q in alternatives
                    ],
                    "ga_alt_steps": [
                        self.vocab.encode(env2.clone().step(q))
                        for q in alternatives
                    ],
                }
                if self.geo_rank_horizon > 1 and self.geo_rank_policy == "greedy":
                    ga.update(
                        ga_greedy=True,
                        ga_problem=fp,
                        ga_trace=list(trace),
                        ga_vocab=self.vocab,
                        ga_env_kind="faithful",
                    )
                elif self.geo_rank_horizon > 1:
                    rollout_steps = []
                    for candidate in candidates:
                        candidate_rollouts = []
                        for _ in range(self.geo_rank_rollouts):
                            roll_env = env2.clone()
                            sequence = list(steps[:t_star])
                            sequence.append(
                                self.vocab.encode(roll_env.step(candidate))
                            )
                            for _depth in range(1, self.geo_rank_horizon):
                                if roll_env.solved:
                                    break
                                feasible = roll_env.feasible_actions()
                                if not feasible:
                                    break
                                nxt = feasible[rng.randrange(len(feasible))]
                                sequence.append(
                                    self.vocab.encode(roll_env.step(nxt))
                                )
                            candidate_rollouts.append(sequence)
                        rollout_steps.append(candidate_rollouts)
                    ga["ga_rollout_steps"] = rollout_steps
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
        if self.macro_alt_k and len(trace) >= self.macro_alt_horizon:
            import numpy as np

            K = self.macro_alt_horizon
            macro_rng = random.Random(f"{self.seed}:{index}:macro-alt")
            t_star = macro_rng.choice(list(range(len(trace) - K + 1)))
            env3 = FaithfulEnv(fp)
            for action in trace[:t_star]:
                env3.step(action)
            candidates = enumerate_faithful_action_sequences(
                env3, K, cap=max(64, 8 * self.macro_alt_k)
            )
            factual = list(trace[t_star:t_star + K])
            alternatives = [seq for seq in candidates if seq != factual]
            macro_rng.shuffle(alternatives)
            chosen = [factual] + alternatives[:self.macro_alt_k]
            macro_actions = []
            macro_steps = []
            macro_remaining = []
            macro_prefix_remaining = []
            before = env3.remaining_necessary()
            py_state = random.getstate()
            np_state = np.random.get_state()
            try:
                for sequence in chosen:
                    clone = env3.clone()
                    future = []
                    prefix_remaining = []
                    for action in sequence:
                        future.append(self.vocab.encode(clone.step(action)))
                        prefix_remaining.append(clone.remaining_necessary())
                    macro_actions.append([
                        self.vocab.encode(env3.action_text(action))
                        for action in sequence
                    ])
                    macro_steps.append(list(steps[:t_star]) + future)
                    macro_remaining.append(clone.remaining_necessary())
                    macro_prefix_remaining.append(prefix_remaining)
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
            out.update(
                macro_alt_t=t_star,
                macro_alt_actions=macro_actions,
                macro_alt_steps=macro_steps,
                macro_alt_remaining=macro_remaining,
                macro_alt_prefix_remaining=macro_prefix_remaining,
                macro_alt_advantage=[
                    before - rem for rem in macro_remaining
                ],
            )
        if self.all_action_supervision:
            out.update(
                action_candidate_tokens=[
                    self.vocab.encode(FaithfulEnv(fp).action_text(action))
                    for action in fp.action_order
                ],
                action_feasible=action_feasible,
            )
        if self.n_alt:
            out["alt_actions"] = alt_actions
            out["alt_steps"] = alt_steps
            out["alt_remaining"] = alt_remaining
        out.update(ga)
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
