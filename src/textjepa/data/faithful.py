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
INVALID_DEFINITION_OUTCOME = (
    "The proposed definition is invalid and nothing changes ."
)


def _fix_seed(key: str) -> None:
    import hashlib

    from tools.tools import fix_seed

    # stable across processes (builtin hash() is salted per process)
    h = int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "little")
    fix_seed(h % (2**31 - 1))


def _necessary_count(g) -> int:
    """Solution length of the query: ancestors of the queried parameter in
    the official dependency graph, plus the query itself (RNG node excluded).
    This is the quantity a planning band should select on — ``n_op`` counts
    ALL operations in the problem, so banding on it near ``max_op`` silently
    removes distractors (op_range=[28,32] with max_op=32 left only ~25% of
    the catalogue unnecessary, making random near-optimal)."""
    import networkx as nx

    p = g.problem
    return len(
        {q for q in nx.ancestors(p.template, p.ques_idx) if q[0] != -1}
        | {p.ques_idx}
    )


def gen_problem(key: str, max_op: int, max_edge: int, op_range=(None, None),
                hash_bins=None, necessary_range=(None, None)):
    """Deterministic official-generator call; optional rejection on n_op
    and/or on the necessary-step count of the query."""
    from data_gen.pretrain.id_gen import IdGen

    _fix_seed(key)
    for attempt in range(200):
        g = IdGen(max_op=max_op, max_edge=max_edge, perm_level=5,
                  detail_level=0)
        g.gen_prob(
            list(range(23)) if hash_bins is None else list(hash_bins),
            p_format="pq",
        )
        lo, hi = op_range
        n = g.problem.n_op
        if not ((lo is None or n >= lo) and (hi is None or n <= hi)):
            continue
        nlo, nhi = necessary_range
        if nlo is not None or nhi is not None:
            k = _necessary_count(g)
            if not ((nlo is None or k >= nlo) and (nhi is None or k <= nhi)):
                continue
        return g
    raise RuntimeError(
        f"gen_problem: no problem in op_range={op_range} "
        f"necessary_range={necessary_range} after 200 attempts (key={key})"
    )


class FaithfulProblem:
    def __init__(self, gen):
        import networkx as nx

        p = gen.problem
        self.p = p
        # IdGen constructs these canonical strings before optionally encoding
        # them with its GPT-2 display tokenizer.  Consume the strings directly:
        # token round-tripping is irrelevant to TextJEPA and can be unavailable
        # (or backed by a fallback codec) on offline clusters.
        self.text = gen.prob.strip()
        self.sol_text = gen.sol.strip()
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
        # reference solutions name parameters WITHOUT the 'each' prefix.
        # Strip ONLY the leading prefix: replace() also deleted the substring
        # inside entity names containing "each " (e.g. "Beach Homes" ->
        # "BHomes"), yielding intent phrases that never occur in prompts or
        # reference solutions (and are out-of-vocabulary, so training targets
        # contained <unk> while free-generation grounding required the exact
        # corrupted string — those actions were permanently ungroundable).
        self.names = {
            q: p.get_param(q).removeprefix("each ") for q in self.params
        }
        self.values = {q: int(p.sketch[q].get_value.a) for q in self.params
                       if q in p.sketch}
        # prompt sentences: the reference text is sentence-per-parameter,
        # final sentence is the question
        parts = [s.strip() + "." for s in self.text.split(". ") if s.strip()]
        parts[-1] = parts[-1].rstrip(".")
        self.prompt_sentences = parts

    def make_env(self) -> "FaithfulEnv":
        """Planning/eval environment for this problem (uniform entry point
        shared with the stylized adapter, see data/stylized_flat.py)."""
        return FaithfulEnv(self)

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

    def step_or_invalid(self, q) -> str:
        """Execute a grounded action without exposing feasibility to a policy."""
        if q not in self.feasible_actions():
            return INVALID_DEFINITION_OUTCOME
        return self.step(q)

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

    INVALID_OUTCOME = INVALID_DEFINITION_OUTCOME

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
        geo_rank_horizons=None,
        geo_rank_rollout_for_h1: bool = False,
        geo_rank_candidate_interface: str = "feasible_menu",
        geo_rank_factual_only: bool = False,
        geo_rank_feasible_k=None,
        geo_rank_invalid_k=None,
        invalid_action_mode: str = "noop",
        geo_rank_rollouts: int = 1,
        geo_rank_policy: str = "random",
        geo_rank_beam_width: int = 1,
        invalid_counterfactual_k: int = 0,
        invalid_counterfactual_unresolved_only: bool = False,
        invalid_counterfactual_resolved_k: int = 0,
        rollout_counterfactual_k: int = 0,
        rollout_solution_prob: float = 0.0,
        macro_alt_k: int = 0,
        macro_alt_horizon: int = 3,
        all_action_supervision: bool = False,
        shuffle_actions: bool = False,
        hash_bins=None,
        necessary_range=(None, None),
        **_,
    ):
        self.shuffle_actions = bool(shuffle_actions)
        self.n_alt = n_alt
        self.geo_rank_k = geo_rank_k
        self.geo_rank_horizon = max(1, int(geo_rank_horizon))
        self.geo_rank_horizons = tuple(
            max(1, int(h)) for h in (geo_rank_horizons or [])
        )
        self.geo_rank_rollout_for_h1 = bool(geo_rank_rollout_for_h1)
        # The faithful adapter implements only the default behavior for the
        # following stylized-dataset knobs.  Accept the defaults so shared
        # launch scripts can pass them, but fail loudly on anything else —
        # silently dropping a control flag is how the shuffle_actions gate
        # defect happened.
        if geo_rank_candidate_interface != "feasible_menu":
            raise NotImplementedError(
                "faithful adapter only supports "
                "geo_rank_candidate_interface=feasible_menu"
            )
        if geo_rank_factual_only:
            raise NotImplementedError(
                "faithful adapter does not implement geo_rank_factual_only"
            )
        if geo_rank_feasible_k is not None or geo_rank_invalid_k is not None:
            raise NotImplementedError(
                "faithful adapter does not implement stratified "
                "geo_rank_feasible_k/geo_rank_invalid_k sampling"
            )
        if invalid_action_mode != "noop":
            raise NotImplementedError(
                "faithful adapter only supports invalid_action_mode=noop"
            )
        self.geo_rank_rollouts = max(1, int(geo_rank_rollouts))
        self.geo_rank_policy = str(geo_rank_policy)
        self.geo_rank_beam_width = max(1, int(geo_rank_beam_width))
        self.invalid_counterfactual_k = max(
            0, int(invalid_counterfactual_k)
        )
        # Hard negatives only: an already-resolved variable is an EASY
        # infeasible (it appears in the step history), while an unresolved
        # variable with unmet prerequisites is the kind planning actually has
        # to reject.  Training the cycle contrast on the easy kind does not
        # transfer to the hard kind (2026-08-15 predcycle AUC stayed at
        # chance while the contrast loss fell below neutral).
        self.invalid_counterfactual_unresolved_only = bool(
            invalid_counterfactual_unresolved_only
        )
        # Easy negatives on top of the hard ones: already-resolved actions
        # (they appear in the step history).  Together with the unresolved
        # hard negatives the energy sees the whole legality boundary.
        self.invalid_counterfactual_resolved_k = max(
            0, int(invalid_counterfactual_resolved_k)
        )
        # Counterfactual (infeasible) intents at every depth of the random
        # rollouts (half premature-unresolved, half already-resolved), so the
        # energy can be contrasted along IMAGINED prefixes, not only at the
        # anchor.  Emitted as ``ga_rollout_cf_actions[c][r][h]`` = token lists
        # of infeasible intents at the rollout state BEFORE rollout action h
        # (h >= 1; index 0 is left empty -- the anchor is covered by
        # ``ga_alt_actions``).
        self.rollout_counterfactual_k = max(0, int(rollout_counterfactual_k))
        # ROLLOUT POLICY (see __getitem__).  Probability that a rollout step
        # follows the environment's own reference solution -- i.e. picks a
        # feasible action that the recorded correct solution actually needs --
        # instead of a uniformly random feasible action.  0.0 reproduces the
        # historical behaviour exactly (no extra RNG draw is made).
        self.rollout_solution_prob = float(rollout_solution_prob)
        if not 0.0 <= self.rollout_solution_prob <= 1.0:
            raise ValueError(
                f"rollout_solution_prob must be in [0, 1]: "
                f"{self.rollout_solution_prob}"
            )
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
        self.hash_bins = None if hash_bins is None else tuple(hash_bins)
        self.necessary_range = tuple(necessary_range)

    def __len__(self) -> int:
        return self.size

    def problem(self, index: int):
        gen = gen_problem(
            f"{self.seed}:{index}", self.max_op, self.max_edge, self.op_range,
            self.hash_bins, self.necessary_range,
        )
        return FaithfulProblem(gen), random.Random(f"{self.seed}:{index}:t")

    def __getitem__(self, index: int) -> dict:
        fp, rng = self.problem(index)
        # Multi-horizon GAR: sample this item's teacher horizon from the
        # configured set on an independent stream, exactly as the stylized
        # dataset does, so enabling the set cannot perturb the trajectory.
        geo_rank_horizon = self.geo_rank_horizon
        if self.geo_rank_horizons:
            horizon_rng = random.Random(f"{self.seed}:{index}:ga-horizon")
            geo_rank_horizon = horizon_rng.choice(self.geo_rank_horizons)
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
            infeasible = [
                q for q in fp.action_order
                if q != executed and q not in env2.feasible_actions()
                and not (
                    self.invalid_counterfactual_unresolved_only
                    and q in env2.resolved
                )
            ]
            rng.shuffle(infeasible)
            alternatives.extend(
                infeasible[: self.invalid_counterfactual_k]
            )
            if self.invalid_counterfactual_resolved_k:
                resolved_neg = [
                    q for q in fp.action_order
                    if q in env2.resolved and q not in alternatives
                ]
                rng.shuffle(resolved_neg)
                alternatives.extend(
                    resolved_neg[: self.invalid_counterfactual_resolved_k]
                )
            if alternatives:
                candidates = [executed, *alternatives]
                ga = {
                    "ga_t": t_star,
                    "ga_horizon": geo_rank_horizon,
                    "ga_beam_width": self.geo_rank_beam_width,
                    "ga_candidate_objects": candidates,
                    "ga_alt_actions": [
                        self.vocab.encode(env2.action_text(q))
                        for q in alternatives
                    ],
                    "ga_alt_steps": [
                        self.vocab.encode(env2.clone().step_or_invalid(q))
                        for q in alternatives
                    ],
                }
                if geo_rank_horizon > 1 and self.geo_rank_policy == "greedy":
                    ga.update(
                        ga_greedy=True,
                        ga_problem=fp,
                        ga_trace=list(trace),
                        ga_vocab=self.vocab,
                        ga_env_kind="faithful",
                    )
                elif geo_rank_horizon > 1 or self.geo_rank_rollout_for_h1:
                    rollout_steps = []
                    rollout_actions = []
                    rollout_cf_actions = []
                    rollout_cf_kinds = []
                    for candidate in candidates:
                        candidate_rollouts = []
                        candidate_action_rollouts = []
                        candidate_cf_rollouts = []
                        candidate_cf_kind_rollouts = []
                        for _ in range(self.geo_rank_rollouts):
                            roll_env = env2.clone()
                            cf_sequence = [[]]
                            cf_kind_sequence = [[]]
                            sequence = list(steps[:t_star])
                            # Horizon-mode GAR consumes the intent phrases of
                            # the rollout actions (ga_rollout_actions ->
                            # ga_rollout_action_tokens); without them the
                            # geo_horizon_rank objective silently skips.
                            action_sequence = [
                                self.vocab.encode(env2.action_text(candidate))
                            ]
                            sequence.append(
                                # ``candidates`` deliberately includes
                                # infeasible actions when invalid
                                # counterfactual supervision is enabled.
                                # Render their executor outcome without
                                # mutating the rollout state; calling
                                # ``step`` here asserted before training.
                                self.vocab.encode(
                                    roll_env.step_or_invalid(candidate)
                                )
                            )
                            for _depth in range(1, geo_rank_horizon):
                                if roll_env.solved:
                                    break
                                feasible = roll_env.feasible_actions()
                                if not feasible:
                                    break
                                if self.rollout_counterfactual_k:
                                    feasible_set = set(feasible)
                                    premature = [
                                        q for q in fp.action_order
                                        if q not in feasible_set
                                        and q not in roll_env.resolved
                                    ]
                                    resolved_neg = [
                                        q for q in fp.action_order
                                        if q in roll_env.resolved
                                    ]
                                    rng.shuffle(premature)
                                    rng.shuffle(resolved_neg)
                                    half = (self.rollout_counterfactual_k + 1) // 2
                                    chosen = premature[:half] + resolved_neg[:half]
                                    # 1 = premature (parents unresolved),
                                    # 2 = already resolved (legal-looking but
                                    # pointless).  Negative-sampling control
                                    # only; never a ranking label.
                                    kinds = (
                                        [1] * len(premature[:half])
                                        + [2] * len(resolved_neg[:half])
                                    )
                                    cf_sequence.append([
                                        self.vocab.encode(roll_env.action_text(q))
                                        for q in chosen[: self.rollout_counterfactual_k]
                                    ])
                                    cf_kind_sequence.append(
                                        kinds[: self.rollout_counterfactual_k]
                                    )
                                # ROLLOUT POLICY.  With probability
                                # ``rollout_solution_prob`` continue along the
                                # environment's reference solution (a feasible
                                # action the correct solution still needs) --
                                # the SAME rule that generates the factual
                                # trajectory above (``rng.choice(nec)``).
                                # Otherwise fall back to a uniformly random
                                # feasible action (historical behaviour).  The
                                # short-circuit keeps prob=0 bit-identical: no
                                # extra draw is taken from ``rng``.
                                nxt = None
                                if (
                                    self.rollout_solution_prob > 0.0
                                    and rng.random() < self.rollout_solution_prob
                                ):
                                    on_path = [
                                        q for q in feasible if q in fp.necessary
                                    ]
                                    if on_path:
                                        nxt = on_path[
                                            rng.randrange(len(on_path))
                                        ]
                                if nxt is None:
                                    nxt = feasible[rng.randrange(len(feasible))]
                                action_sequence.append(
                                    self.vocab.encode(
                                        roll_env.action_text(nxt)
                                    )
                                )
                                sequence.append(
                                    self.vocab.encode(roll_env.step(nxt))
                                )
                            candidate_rollouts.append(sequence)
                            candidate_action_rollouts.append(action_sequence)
                            candidate_cf_rollouts.append(cf_sequence)
                            candidate_cf_kind_rollouts.append(cf_kind_sequence)
                        rollout_steps.append(candidate_rollouts)
                        rollout_actions.append(candidate_action_rollouts)
                        rollout_cf_actions.append(candidate_cf_rollouts)
                        rollout_cf_kinds.append(candidate_cf_kind_rollouts)
                    ga["ga_rollout_steps"] = rollout_steps
                    ga["ga_rollout_actions"] = rollout_actions
                    if self.rollout_counterfactual_k:
                        ga["ga_rollout_cf_actions"] = rollout_cf_actions
                        ga["ga_rollout_cf_kinds"] = rollout_cf_kinds
        # Grounding falsifier (named negative control): permute the
        # correspondence between on-trajectory action phrases and their
        # rendered transitions.  Drawn last, after all other randomness, to
        # keep every other field identical to the aligned condition — same
        # contract as the stylized dataset.
        if self.shuffle_actions and len(actions) > 1:
            rng.shuffle(actions)
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
    words.update(INVALID_DEFINITION_OUTCOME.split())
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


_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
_VOCAB_CACHE = _CONFIGS_DIR / "faithful_vocab.txt"


def cached_faithful_vocab(max_op: int = 21, max_edge: int = 28) -> Vocab:
    """Vocab scanned at the given generator caps.

    The historical cache ``faithful_vocab.txt`` was scanned at 21/28 (its
    token list is frozen — every existing checkpoint's embedding table is
    indexed by it).  Other caps use a separate ``faithful_vocab_{op}_{edge}``
    cache so eval-band names (e.g. at 32/40) are in-vocabulary."""
    cache = (
        _VOCAB_CACHE if (max_op, max_edge) == (21, 28)
        else _CONFIGS_DIR / f"faithful_vocab_{max_op}_{max_edge}.txt"
    )
    if cache.exists():
        return Vocab(cache.read_text().split("\n"))
    v = build_faithful_vocab(max_op=max_op, max_edge=max_edge)
    cache.write_text(
        "\n".join(t for t in v.token_to_id if not t.startswith("<"))
    )
    return v
