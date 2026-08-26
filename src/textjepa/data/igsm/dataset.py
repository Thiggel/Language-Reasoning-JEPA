"""Torch dataset of iGSM reasoning traces with ground-truth probe labels.

Problems are generated on the fly, deterministically per (seed, index).
Traces follow a mildly suboptimal policy: mostly necessary steps with
occasional distractor resolutions, so that value/goal heads see off-path
states and planners face a real "which step matters" choice.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import CONST_OP, OPS, Problem, sample_problem
from textjepa.data.igsm.render import TEMPLATE_WORDS, action_phrase, prompt_sentences
from textjepa.data.vocab import EDIT_WORDS, Vocab

OP_LABELS = {CONST_OP: 0, **{op: i + 1 for i, op in enumerate(OPS)}}

DEFAULT_ADJECTIVES = [
    "red", "blue", "green", "yellow", "purple", "orange", "silver", "golden",
    "small", "large", "old", "new", "round", "square", "heavy", "light",
    "shiny", "dark", "soft", "hard",
]
DEFAULT_NOUNS = [
    "apples", "keys", "pens", "boxes", "books", "coins", "cups", "hats",
    "stones", "cards", "shells", "beads", "nails", "ropes", "jars", "bells",
    "lamps", "forks", "tiles", "knots",
]


def build_vocab(modulus: int, adjectives=None, nouns=None) -> Vocab:
    adjectives = adjectives or DEFAULT_ADJECTIVES
    nouns = nouns or DEFAULT_NOUNS
    tokens = list(TEMPLATE_WORDS) + list(EDIT_WORDS) + adjectives + nouns
    tokens += [str(i) for i in range(modulus)]
    return Vocab(tokens)


def rollout_trace(
    p: Problem, rng: random.Random, distractor_prob: float, max_distractors: int
) -> list[int]:
    """Action sequence solving ``p`` with some distractor detours."""
    env = SymbolicEnv(p)
    trace: list[int] = []
    n_distractors = 0
    while not env.solved:
        feasible = env.feasible_actions()
        necessary = [i for i in feasible if i in p.query_ancestors]
        distractors = [i for i in feasible if i not in p.query_ancestors]
        use_distractor = (
            distractors
            and n_distractors < max_distractors
            and rng.random() < distractor_prob
        )
        pick = rng.choice(distractors) if use_distractor else rng.choice(necessary)
        n_distractors += int(pick not in p.query_ancestors)
        env.step(pick)
        trace.append(pick)
    return trace


def enumerate_action_sequences(
    p: Problem,
    resolved: frozenset[int],
    depth: int,
    cap: int = 256,
) -> list[list[int]]:
    """Enumerate fixed-depth feasible chunks for macro counterfactuals."""
    frontier: list[tuple[list[int], frozenset[int]]] = [([], resolved)]
    for _ in range(depth):
        nxt = []
        for seq, done in frontier:
            feasible = [
                v.idx for v in p.vars
                if v.idx not in done and all(pa in done for pa in v.parents)
            ]
            for action in feasible:
                nxt.append((seq + [action], done | {action}))
        frontier = nxt[:cap]
        if not frontier:
            break
    return [seq for seq, _ in frontier if len(seq) == depth]


class IGSMDataset(Dataset):
    def __init__(
        self,
        vocab: Vocab,
        size: int,
        seed: int,
        modulus: int = 23,
        n_vars_range: tuple[int, int] = (6, 12),
        leaf_prob: float = 0.35,
        steps_range: tuple[int, int] = (3, 9),
        distractor_prob: float = 0.15,
        max_distractors: int = 2,
        shuffle_actions: bool = False,
        n_alt: int = 0,
        geo_rank_k: int = 0,
        geo_rank_factual_only: bool = False,
        geo_rank_horizon: int = 1,
        geo_rank_horizons: list[int] | None = None,
        geo_rank_rollouts: int = 1,
        geo_rank_rollout_for_h1: bool = False,
        geo_rank_policy: str = "random",
        geo_rank_beam_width: int = 1,
        geo_rank_candidate_interface: str = "feasible_menu",
        geo_rank_feasible_k: int | None = None,
        geo_rank_invalid_k: int | None = None,
        invalid_action_mode: str = "noop",
        macro_alt_k: int = 0,
        macro_alt_horizon: int = 3,
        all_action_supervision: bool = False,
        invalid_counterfactual_k: int = 0,
        invalid_counterfactual_unresolved_only: bool = False,
        invalid_counterfactual_resolved_k: int = 0,
        rollout_counterfactual_k: int = 0,
        rollout_solution_prob: float = 0.0,
        depth_uniform_anchors: bool = False,
        hindsight_long_rollout: bool = False,
        hindsight_long_max=None,
        sample_max_tries: int = 50,
        strict_steps_range: bool = False,
        adjectives: list[str] | None = None,
        nouns: list[str] | None = None,
    ):
        self.vocab = vocab
        self.size = size
        self.seed = seed
        self.modulus = modulus
        self.n_vars_range = tuple(n_vars_range)
        self.leaf_prob = leaf_prob
        self.steps_range = tuple(steps_range)
        self.distractor_prob = distractor_prob
        self.max_distractors = max_distractors
        self.shuffle_actions = shuffle_actions  # control: break action grounding
        self.n_alt = n_alt  # counterfactual candidates per step (ranking)
        self.geo_rank_k = geo_rank_k  # geometric-advantage ranking anchors
        # K=0 cell: anchor + rollouts with the factual root only (no
        # alternative actions anywhere in the energy supervision)
        self.geo_rank_factual_only = bool(geo_rank_factual_only)
        self.geo_rank_horizon = max(1, int(geo_rank_horizon))
        self.geo_rank_horizons = tuple(
            max(1, int(horizon)) for horizon in (geo_rank_horizons or [])
        )
        self.geo_rank_rollouts = max(1, int(geo_rank_rollouts))
        self.geo_rank_rollout_for_h1 = bool(geo_rank_rollout_for_h1)
        self.geo_rank_policy = str(geo_rank_policy)
        self.geo_rank_beam_width = max(1, int(geo_rank_beam_width))
        self.geo_rank_candidate_interface = str(
            geo_rank_candidate_interface
        )
        self.geo_rank_feasible_k = geo_rank_feasible_k
        self.geo_rank_invalid_k = geo_rank_invalid_k
        self.invalid_action_mode = str(invalid_action_mode)
        self.macro_alt_k = max(0, int(macro_alt_k))
        self.macro_alt_horizon = max(1, int(macro_alt_horizon))
        self.all_action_supervision = bool(all_action_supervision)
        # Hard negatives for the energy feasibility ranking (same recipe as
        # FaithfulDataset): premature intents (unresolved, parents missing)
        # at the ranking anchor, optional already-resolved intents on top,
        # and infeasible intents at every imagined rollout depth
        # (``ga_rollout_cf_actions[c][r][h]`` = token lists).
        self.invalid_counterfactual_k = max(0, int(invalid_counterfactual_k))
        self.invalid_counterfactual_unresolved_only = bool(
            invalid_counterfactual_unresolved_only
        )
        self.invalid_counterfactual_resolved_k = max(
            0, int(invalid_counterfactual_resolved_k)
        )
        self.rollout_counterfactual_k = max(0, int(rollout_counterfactual_k))
        # See FaithfulDataset.rollout_solution_prob: probability that a
        # rollout step follows the reference solution instead of a uniformly
        # random feasible action.  0.0 = historical behaviour, bit-identical.
        self.rollout_solution_prob = float(rollout_solution_prob)
        if not 0.0 <= self.rollout_solution_prob <= 1.0:
            raise ValueError(
                f"rollout_solution_prob must be in [0, 1]: "
                f"{self.rollout_solution_prob}"
            )
        # See FaithfulDataset: sample the ranking anchor's steps-to-go
        # uniformly up to the generator cap (far-from-goal contrasts as
        # frequent as near-goal ones); off = historical, bit-identical.
        self.depth_uniform_anchors = bool(depth_uniform_anchors)
        # See FaithfulDataset: extra random feasible rollouts from the
        # anchor with horizon up to the FULL remaining trajectory length,
        # for hindsight_long_horizon_rank.  Isolated RNG stream.
        self.hindsight_long_rollout = bool(hindsight_long_rollout)
        self.hindsight_long_max = (
            None if hindsight_long_max is None
            else max(1, int(hindsight_long_max))
        )
        self.sample_max_tries = max(1, int(sample_max_tries))
        self.strict_steps_range = bool(strict_steps_range)
        if self.geo_rank_policy not in {"random", "greedy", "latent_beam"}:
            raise ValueError(f"unknown geo_rank_policy: {self.geo_rank_policy}")
        if self.geo_rank_candidate_interface not in {
            "feasible_menu", "full_catalogue"
        }:
            raise ValueError(
                "unknown geometric-ranking candidate interface: "
                f"{self.geo_rank_candidate_interface}"
            )
        if self.invalid_action_mode not in {"noop", "failure"}:
            raise ValueError(
                f"unknown invalid action mode: {self.invalid_action_mode}"
            )
        self.adjectives = adjectives or DEFAULT_ADJECTIVES
        self.nouns = nouns or DEFAULT_NOUNS

    def __len__(self) -> int:
        return self.size

    def problem(self, index: int) -> tuple[Problem, random.Random]:
        rng = random.Random(f"{self.seed}:{index}")
        p = sample_problem(
            rng,
            self.adjectives,
            self.nouns,
            self.modulus,
            self.n_vars_range,
            self.leaf_prob,
            self.steps_range,
            max_tries=self.sample_max_tries,
        )
        if self.strict_steps_range and not (
            self.steps_range[0]
            <= p.n_necessary_steps
            <= self.steps_range[1]
        ):
            raise RuntimeError(
                "failed to sample a problem in the requested exact reasoning "
                f"length range {self.steps_range} after "
                f"{self.sample_max_tries} attempts"
            )
        return p, rng

    def __getitem__(self, index: int) -> dict:
        p, rng = self.problem(index)
        geo_rank_horizon = self.geo_rank_horizon
        if self.geo_rank_horizons:
            horizon_rng = random.Random(f"{self.seed}:{index}:ga-horizon")
            geo_rank_horizon = horizon_rng.choice(self.geo_rank_horizons)
        # Counterfactual-set sampling is an optional supervision view.  Give it
        # an independent stream so enabling n_alt cannot change the trajectory,
        # geometric teacher, or grounding-control permutation.
        alt_rng = random.Random(f"{self.seed}:{index}:alt")
        trace = rollout_trace(p, rng, self.distractor_prob, self.max_distractors)

        prompt = [self.vocab.encode(s) for s in prompt_sentences(p, rng)]
        env = SymbolicEnv(p)
        steps, actions, op, value, remaining, resolved_n, necessary = (
            [], [], [], [], [], [], []
        )
        alt_actions: list[list[list[int]]] = []
        alt_steps: list[list[list[int]]] = []
        alt_remaining: list[list[int]] = []
        action_feasible: list[list[int]] = []
        for idx in trace:
            actions.append(self.vocab.encode(action_phrase(p, idx)))
            if self.all_action_supervision:
                feasible_set = set(env.feasible_actions())
                action_feasible.append([
                    int(variable.idx in feasible_set) for variable in p.vars
                ])
            if self.n_alt:
                done = env.resolved_set
                others = [a for a in env.feasible_actions() if a != idx]
                alt_rng.shuffle(others)
                alts = others[: self.n_alt]
                alt_actions.append(
                    [self.vocab.encode(action_phrase(p, a)) for a in alts]
                )
                alt_steps.append(
                    [self.vocab.encode(env.clone().step(a)) for a in alts]
                )
                alt_remaining.append(
                    [len(p.query_ancestors - (done | {a})) for a in alts]
                )
            steps.append(self.vocab.encode(env.step(idx)))
            v = p.vars[idx]
            op.append(OP_LABELS[v.op])
            value.append(p.values[idx])
            remaining.append(env.remaining_necessary())
            resolved_n.append(len(env.resolved))
            necessary.append(int(idx in p.query_ancestors))
        ga = {}
        if (self.geo_rank_k or self.geo_rank_factual_only) and len(trace) > 1:
            # one anchor step: alt intent phrases + env-rendered TRUE next
            # step sentences (text only; the ranking label is computed in
            # latent space by the model — no symbolic annotations)
            if self.depth_uniform_anchors:
                # Steps-to-go uniform up to the generator cap; the clamp
                # piles the excess on the trajectory start, the farthest
                # state this trace has (see __init__).
                cap = self.steps_range[1] + self.max_distractors
                d = 1 + rng.randrange(max(cap, 1))
                t_star = len(trace) - min(d, len(trace))
            else:
                t_star = rng.randrange(len(trace))
            env2 = SymbolicEnv(p, self.invalid_action_mode)
            for i in trace[:t_star]:
                env2.step(i)
            if self.geo_rank_candidate_interface == "full_catalogue":
                feasible = set(env2.feasible_actions())
                valid_others = [
                    action for action in feasible
                    if action != trace[t_star]
                ]
                invalid_others = [
                    variable.idx for variable in p.vars
                    if variable.idx not in feasible
                ]
                rng.shuffle(valid_others)
                rng.shuffle(invalid_others)
                if (
                    self.geo_rank_feasible_k is not None
                    or self.geo_rank_invalid_k is not None
                ):
                    others = (
                        valid_others[: self.geo_rank_feasible_k or 0]
                        + invalid_others[: self.geo_rank_invalid_k or 0]
                    )
                else:
                    others = valid_others + invalid_others
            else:
                others = [
                    a for a in env2.feasible_actions() if a != trace[t_star]
                ]
            rng.shuffle(others)
            alts = (
                others if self.geo_rank_k < 0
                else others[: self.geo_rank_k]
            )
            if self.geo_rank_factual_only:
                alts = []
            or_invalid = (
                self.geo_rank_candidate_interface == "full_catalogue"
                or bool(self.invalid_counterfactual_k
                        or self.invalid_counterfactual_resolved_k)
            )
            if self.invalid_counterfactual_k or self.invalid_counterfactual_resolved_k:
                feasible_now = set(env2.feasible_actions())
                resolved_now = env2.resolved_set
                chosen = set(alts) | {trace[t_star]}
                infeasible = [
                    v.idx for v in p.vars
                    if v.idx not in chosen and v.idx not in feasible_now
                    and not (
                        self.invalid_counterfactual_unresolved_only
                        and v.idx in resolved_now
                    )
                ]
                rng.shuffle(infeasible)
                alts = list(alts) + infeasible[: self.invalid_counterfactual_k]
                if self.invalid_counterfactual_resolved_k:
                    chosen = set(alts) | {trace[t_star]}
                    resolved_neg = [
                        v.idx for v in p.vars
                        if v.idx in resolved_now and v.idx not in chosen
                    ]
                    rng.shuffle(resolved_neg)
                    alts = alts + resolved_neg[
                        : self.invalid_counterfactual_resolved_k
                    ]
            if alts or self.geo_rank_factual_only:
                ga = {
                    "ga_t": t_star,
                    "ga_horizon": geo_rank_horizon,
                    "ga_beam_width": self.geo_rank_beam_width,
                    "ga_candidate_ids": [trace[t_star], *alts],
                    "ga_candidate_objects": [trace[t_star], *alts],
                    "ga_alt_actions": [
                        self.vocab.encode(action_phrase(p, a)) for a in alts
                    ],
                    "ga_alt_steps": [
                        self.vocab.encode(
                            env2.clone().step_or_invalid(a) if or_invalid
                            else env2.clone().step(a)
                        )
                        for a in alts
                    ],
                }
                if (
                    geo_rank_horizon > 1
                    and self.geo_rank_policy in {"greedy", "latent_beam"}
                ):
                    # The model follows the greedy continuation online because
                    # the policy depends on the current EMA geometry.  Keep the
                    # symbolic problem only as an interaction interface; no
                    # ancestor, remaining-step, or preference labels are used.
                    ga.update(
                        ga_greedy=self.geo_rank_policy == "greedy",
                        ga_latent_beam=self.geo_rank_policy == "latent_beam",
                        ga_problem=p,
                        ga_trace=list(trace),
                        ga_vocab=self.vocab,
                        ga_env_kind="stylized",
                    )
                elif geo_rank_horizon > 1 or self.geo_rank_rollout_for_h1:
                    # Monte-Carlo shooting approximation to an N-step optimal
                    # continuation.  The dataset supplies only feasible action
                    # interactions and rendered text; the model later selects
                    # the rollout with minimum EMA latent goal distance.  No
                    # remaining-step or relevance labels enter that selection.
                    candidates = [trace[t_star], *alts]
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
                            action_sequence = [
                                self.vocab.encode(action_phrase(p, candidate))
                            ]
                            outcome = (
                                roll_env.step_or_invalid(candidate)
                                if or_invalid else roll_env.step(candidate)
                            )
                            sequence.append(self.vocab.encode(outcome))
                            for _depth in range(1, geo_rank_horizon):
                                if roll_env.solved:
                                    break
                                if roll_env.failed:
                                    # Supply absorbing-failure transitions so
                                    # imagined search cannot escape the state.
                                    nxt = rng.randrange(len(p.vars))
                                    action_sequence.append(
                                        self.vocab.encode(action_phrase(p, nxt))
                                    )
                                    sequence.append(self.vocab.encode(
                                        roll_env.step_or_invalid(nxt)
                                    ))
                                    continue
                                feasible = roll_env.feasible_actions()
                                if not feasible:
                                    break
                                if self.rollout_counterfactual_k:
                                    feasible_set = set(feasible)
                                    resolved_here = roll_env.resolved_set
                                    premature = [
                                        v.idx for v in p.vars
                                        if v.idx not in feasible_set
                                        and v.idx not in resolved_here
                                    ]
                                    resolved_neg = [
                                        v.idx for v in p.vars
                                        if v.idx in resolved_here
                                    ]
                                    rng.shuffle(premature)
                                    rng.shuffle(resolved_neg)
                                    half = (self.rollout_counterfactual_k + 1) // 2
                                    picked = premature[:half] + resolved_neg[:half]
                                    # 1 = premature (parents unresolved),
                                    # 2 = already resolved (legal-looking but
                                    # pointless).  Negative-sampling control
                                    # only; never a ranking label.
                                    kinds = (
                                        [1] * len(premature[:half])
                                        + [2] * len(resolved_neg[:half])
                                    )
                                    cf_sequence.append([
                                        self.vocab.encode(action_phrase(p, q))
                                        for q in picked[
                                            : self.rollout_counterfactual_k
                                        ]
                                    ])
                                    cf_kind_sequence.append(
                                        kinds[: self.rollout_counterfactual_k]
                                    )
                                # ROLLOUT POLICY: follow the reference
                                # solution (feasible query-ancestor steps --
                                # the same rule the factual trajectory uses)
                                # with probability rollout_solution_prob,
                                # else uniformly random feasible.  The
                                # short-circuit keeps prob=0 bit-identical.
                                nxt = None
                                if (
                                    self.rollout_solution_prob > 0.0
                                    and rng.random() < self.rollout_solution_prob
                                ):
                                    on_path = [
                                        i for i in feasible
                                        if i in p.query_ancestors
                                    ]
                                    if on_path:
                                        nxt = on_path[rng.randrange(len(on_path))]
                                if nxt is None:
                                    nxt = feasible[rng.randrange(len(feasible))]
                                action_sequence.append(
                                    self.vocab.encode(action_phrase(p, nxt))
                                )
                                sequence.append(self.vocab.encode(roll_env.step(nxt)))
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

        # HINDSIGHT LONG-HORIZON negatives (see __init__): random feasible
        # rollouts from the anchor state, horizon h uniform up to the full
        # remaining trajectory length.  Isolated RNG stream.
        if self.hindsight_long_rollout and ga:
            hl_rng = random.Random(f"{self.seed}:{index}:hl")
            cap = len(trace) - t_star
            if self.hindsight_long_max is not None:
                cap = min(cap, self.hindsight_long_max)
            h = 1 + hl_rng.randrange(max(cap, 1))
            hl_actions = []
            for _ in range(self.geo_rank_rollouts):
                roll_env = env2.clone()
                seq = []
                for _depth in range(h):
                    if roll_env.solved:
                        break
                    feasible = roll_env.feasible_actions()
                    if not feasible:
                        break
                    nxt = feasible[hl_rng.randrange(len(feasible))]
                    seq.append(self.vocab.encode(action_phrase(p, nxt)))
                    roll_env.step(nxt)
                hl_actions.append(seq)
            ga["ga_hl_h"] = h
            ga["ga_hl_actions"] = hl_actions

        # Keep the grounding falsifier exactly paired with the aligned
        # condition.  In particular, draw the GAR anchor, alternatives, and
        # continuations before consuming randomness for this permutation.
        # The only changed training field is then the correspondence between
        # on-trajectory action phrases and their rendered transitions.
        if self.shuffle_actions and len(actions) > 1:
            rng.shuffle(actions)

        out = {
            "prompt": prompt,
            "steps": steps,
            "actions": actions,
            "op": op,
            "value": value,
            "remaining": remaining,
            "resolved_n": resolved_n,
            "necessary": necessary,
            "answer": p.answer,
            "n_necessary": p.n_necessary_steps,
            "n_vars": len(p.vars),
            "index": index,
            "var_idx": list(trace),  # which variable each step resolved
            "query_idx": p.query,
            "ancestors": sorted(p.query_ancestors),
        }
        if self.macro_alt_k and len(trace) >= self.macro_alt_horizon:
            K = self.macro_alt_horizon
            macro_rng = random.Random(f"{self.seed}:{index}:macro-alt")
            anchors = list(range(len(trace) - K + 1))
            t_star = macro_rng.choice(anchors)
            env3 = SymbolicEnv(p)
            for action in trace[:t_star]:
                env3.step(action)
            candidates = enumerate_action_sequences(
                p,
                frozenset(env3.resolved_set),
                K,
                cap=max(64, 8 * self.macro_alt_k),
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
            for sequence in chosen:
                clone = env3.clone()
                future = []
                prefix_remaining = []
                for action in sequence:
                    future.append(self.vocab.encode(clone.step(action)))
                    prefix_remaining.append(clone.remaining_necessary())
                macro_actions.append([
                    self.vocab.encode(action_phrase(p, a)) for a in sequence
                ])
                macro_steps.append(list(steps[:t_star]) + future)
                macro_remaining.append(clone.remaining_necessary())
                macro_prefix_remaining.append(prefix_remaining)
            out.update(
                macro_alt_t=t_star,
                macro_alt_actions=macro_actions,
                macro_alt_steps=macro_steps,
                macro_alt_remaining=macro_remaining,
                macro_alt_prefix_remaining=macro_prefix_remaining,
                macro_alt_advantage=[before - rem for rem in macro_remaining],
            )
        if self.all_action_supervision:
            out.update(
                action_candidate_tokens=[
                    self.vocab.encode(action_phrase(p, variable.idx))
                    for variable in p.vars
                ],
                action_feasible=action_feasible,
                action_indices=list(trace),
            )
        if self.n_alt:
            out["alt_actions"] = alt_actions
            out["alt_steps"] = alt_steps
            out["alt_remaining"] = alt_remaining
        out.update(ga)
        return out


def _pad_chunks(
    seqs: list[list[list[int]]], pad: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of chunk lists to [B, C, L]; returns (tokens, chunk_mask)."""
    B = len(seqs)
    C = max(len(s) for s in seqs)
    L = max((len(c) for s in seqs for c in s), default=1)
    tokens = torch.full((B, C, L), pad, dtype=torch.long)
    mask = torch.zeros(B, C, dtype=torch.bool)
    for b, s in enumerate(seqs):
        for c, chunk in enumerate(s):
            tokens[b, c, : len(chunk)] = torch.tensor(chunk)
            mask[b, c] = True
    return tokens, mask


def _pad_labels(seqs: list[list[int]], fill: int = 0) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    out = torch.full((len(seqs), T), fill, dtype=torch.long)
    for b, s in enumerate(seqs):
        out[b, : len(s)] = torch.tensor(s)
    return out


def _pad_alt(batch: list[dict], pad: int) -> dict:
    """Pad per-step alternative actions to alt_tokens [B, T, K, L] and
    alt_remaining [B, T, K] (-1 marks absent candidates)."""
    B = len(batch)
    T = max(len(b["alt_actions"]) for b in batch)
    K = max((len(step) for b in batch for step in b["alt_actions"]), default=1)
    K = max(K, 1)
    L = max(
        (len(a) for b in batch for step in b["alt_actions"] for a in step),
        default=1,
    )
    Ls = max(
        (len(s) for b in batch for step in b.get("alt_steps", []) for s in step),
        default=1,
    )
    tokens = torch.full((B, T, K, L), pad, dtype=torch.long)
    steps = torch.full((B, T, K, Ls), pad, dtype=torch.long)
    remaining = torch.full((B, T, K), -1, dtype=torch.long)
    for b, item in enumerate(batch):
        for t, (alts, outcomes, rems) in enumerate(
            zip(item["alt_actions"], item.get("alt_steps", [[]] * T),
                item["alt_remaining"])
        ):
            for k, (a, r) in enumerate(zip(alts, rems)):
                tokens[b, t, k, : len(a)] = torch.tensor(a)
                if k < len(outcomes):
                    steps[b, t, k, : len(outcomes[k])] = torch.tensor(outcomes[k])
                remaining[b, t, k] = r
    return {
        "alt_tokens": tokens,
        "alt_step_tokens": steps,
        "alt_remaining": remaining,
    }


def collate(batch: list[dict], pad_id: int) -> dict:
    prompt_tokens, prompt_mask = _pad_chunks([b["prompt"] for b in batch], pad_id)
    step_tokens, step_mask = _pad_chunks([b["steps"] for b in batch], pad_id)
    action_tokens, _ = _pad_chunks([b["actions"] for b in batch], pad_id)
    extra = _pad_alt(batch, pad_id) if "alt_actions" in batch[0] else {}
    if any("macro_alt_actions" in item for item in batch):
        B = len(batch)
        A = max(
            (len(item.get("macro_alt_actions", [])) for item in batch),
            default=1,
        )
        K = max(
            (len(seq) for item in batch
             for seq in item.get("macro_alt_actions", [])),
            default=1,
        )
        La = max(
            (len(chunk) for item in batch
             for seq in item.get("macro_alt_actions", []) for chunk in seq),
            default=1,
        )
        Tm = max(
            (len(seq) for item in batch
             for seq in item.get("macro_alt_steps", [])),
            default=1,
        )
        Ls = max(
            (len(chunk) for item in batch
             for seq in item.get("macro_alt_steps", []) for chunk in seq),
            default=1,
        )
        mat = torch.full((B, A, K, La), pad_id, dtype=torch.long)
        mst = torch.full((B, A, Tm, Ls), pad_id, dtype=torch.long)
        msm = torch.zeros(B, A, Tm, dtype=torch.bool)
        mav = torch.zeros(B, A, dtype=torch.bool)
        mar = torch.zeros(B, A, dtype=torch.float)
        mapr = torch.zeros(B, A, K, dtype=torch.float)
        maa = torch.zeros(B, A, dtype=torch.float)
        mat_idx = torch.full((B,), -1, dtype=torch.long)
        for b, item in enumerate(batch):
            if "macro_alt_actions" not in item:
                continue
            mat_idx[b] = item["macro_alt_t"]
            for a, (action_seq, step_seq, remaining, advantage) in enumerate(zip(
                item["macro_alt_actions"],
                item["macro_alt_steps"],
                item["macro_alt_remaining"],
                item["macro_alt_advantage"],
            )):
                mav[b, a] = True
                mar[b, a] = remaining
                maa[b, a] = advantage
                prefix_remaining = item["macro_alt_prefix_remaining"][a]
                mapr[b, a, :len(prefix_remaining)] = torch.tensor(
                    prefix_remaining, dtype=torch.float
                )
                for k, chunk in enumerate(action_seq):
                    mat[b, a, k, :len(chunk)] = torch.tensor(chunk)
                for t, chunk in enumerate(step_seq):
                    mst[b, a, t, :len(chunk)] = torch.tensor(chunk)
                    msm[b, a, t] = True
        extra.update(
            macro_alt_t=mat_idx,
            macro_alt_action_tokens=mat,
            macro_alt_step_tokens=mst,
            macro_alt_step_mask=msm,
            macro_alt_valid=mav,
            macro_alt_remaining=mar,
            macro_alt_prefix_remaining=mapr,
            macro_alt_advantage=maa,
        )
    if any("action_candidate_tokens" in item for item in batch):
        B = len(batch)
        V = max(len(item.get("action_candidate_tokens", [])) for item in batch)
        L = max(
            (len(action) for item in batch
             for action in item.get("action_candidate_tokens", [])),
            default=1,
        )
        T = max(len(item.get("action_feasible", [])) for item in batch)
        candidate_tokens = torch.full((B, V, L), pad_id, dtype=torch.long)
        candidate_mask = torch.zeros(B, V, dtype=torch.bool)
        feasible = torch.zeros(B, T, V, dtype=torch.bool)
        candidate_observed = torch.zeros(B, T, V, dtype=torch.bool)
        action_indices = torch.full((B, T), -1, dtype=torch.long)
        for b, item in enumerate(batch):
            for v, action in enumerate(item.get("action_candidate_tokens", [])):
                candidate_tokens[b, v, :len(action)] = torch.tensor(action)
                candidate_mask[b, v] = True
            labels = item.get("action_feasible", [])
            if labels:
                tensor = torch.tensor(labels, dtype=torch.bool)
                feasible[b, :tensor.shape[0], :tensor.shape[1]] = tensor
            indices = item.get("action_indices", [])
            if indices:
                action_indices[b, :len(indices)] = torch.tensor(indices)
            observed = item.get("action_candidate_observed")
            if observed is None:
                candidate_observed[
                    b, :, :len(item.get("action_candidate_tokens", []))
                ] = True
            elif observed:
                tensor = torch.tensor(observed, dtype=torch.bool)
                candidate_observed[
                    b, :tensor.shape[0], :tensor.shape[1]
                ] = tensor
        extra.update(
            action_candidate_tokens=candidate_tokens,
            action_candidate_mask=candidate_mask,
            action_candidate_observed=candidate_observed,
            action_feasible=feasible,
            action_indices=action_indices,
        )
    if any("ga_t" in b for b in batch):
        K = max((len(b.get("ga_alt_actions", [])) for b in batch), default=1)
        La = max((len(x) for b in batch for x in b.get("ga_alt_actions", [])),
                 default=1)
        Ls = max((len(x) for b in batch for x in b.get("ga_alt_steps", [])),
                 default=1)
        B = len(batch)
        gaa = torch.full((B, K, La), pad_id, dtype=torch.long)
        gas = torch.full((B, K, Ls), pad_id, dtype=torch.long)
        gav = torch.zeros(B, K, dtype=torch.bool)
        gat = torch.full((B,), -1, dtype=torch.long)
        gac = torch.full((B, K + 1), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            if "ga_t" not in b:
                continue
            gat[i] = b["ga_t"]
            ids = b.get("ga_candidate_ids", [])
            if ids:
                gac[i, : len(ids)] = torch.tensor(ids)
            for k, (a, st) in enumerate(zip(b["ga_alt_actions"], b["ga_alt_steps"])):
                gaa[i, k, : len(a)] = torch.tensor(a)
                gas[i, k, : len(st)] = torch.tensor(st)
                gav[i, k] = True
        extra.update(ga_t=gat,
                     ga_horizon=max((b.get("ga_horizon", 1) for b in batch)),
                     ga_requested_horizon=torch.tensor([
                         b.get("ga_horizon", 1) for b in batch
                     ], dtype=torch.long),
                     ga_beam_width=max(
                         (b.get("ga_beam_width", 1) for b in batch)
                     ),
                     ga_candidate_ids=gac,
                     ga_alt_action_tokens=gaa,
                     ga_alt_step_tokens=gas, ga_valid=gav)
        if any(
            b.get("ga_greedy", False) or b.get("ga_latent_beam", False)
            for b in batch
        ):
            candidate_objects = [
                list(b.get("ga_candidate_objects", [])) for b in batch
            ]
            max_candidates = K + 1
            candidate_objects = [
                row + [None] * (max_candidates - len(row))
                for row in candidate_objects
            ]
            extra.update(
                ga_greedy=any(b.get("ga_greedy", False) for b in batch),
                ga_latent_beam=any(
                    b.get("ga_latent_beam", False) for b in batch
                ),
                ga_problems=[b.get("ga_problem") for b in batch],
                ga_traces=[b.get("ga_trace") for b in batch],
                ga_candidate_objects=candidate_objects,
                ga_env_kinds=[b.get("ga_env_kind", "stylized") for b in batch],
                ga_vocab=next(b["ga_vocab"] for b in batch if "ga_vocab" in b),
            )
        if any("ga_rollout_steps" in b for b in batch):
            C = max(
                (len(b.get("ga_rollout_steps", [])) for b in batch), default=1
            )
            R = max(
                (len(c) for b in batch for c in b.get("ga_rollout_steps", [])),
                default=1,
            )
            Tr = max(
                (len(seq) for b in batch
                 for c in b.get("ga_rollout_steps", []) for seq in c),
                default=1,
            )
            Lr = max(
                (len(sent) for b in batch
                 for c in b.get("ga_rollout_steps", []) for seq in c
                 for sent in seq),
                default=1,
            )
            grt = torch.full((B, C, R, Tr, Lr), pad_id, dtype=torch.long)
            grm = torch.zeros(B, C, R, Tr, dtype=torch.bool)
            grv = torch.zeros(B, C, R, dtype=torch.bool)
            for i, b in enumerate(batch):
                for c, candidate in enumerate(b.get("ga_rollout_steps", [])):
                    for r, seq in enumerate(candidate):
                        grv[i, c, r] = True
                        for t, sent in enumerate(seq):
                            grt[i, c, r, t, : len(sent)] = torch.tensor(sent)
                            grm[i, c, r, t] = True
            extra.update(
                ga_rollout_step_tokens=grt,
                ga_rollout_step_mask=grm,
                ga_rollout_valid=grv,
            )
            if any("ga_rollout_actions" in b for b in batch):
                Ha = max(
                    (len(seq) for b in batch
                     for c in b.get("ga_rollout_actions", []) for seq in c),
                    default=1,
                )
                La = max(
                    (len(action) for b in batch
                     for c in b.get("ga_rollout_actions", []) for seq in c
                     for action in seq),
                    default=1,
                )
                gra = torch.full(
                    (B, C, R, Ha, La), pad_id, dtype=torch.long
                )
                gram = torch.zeros(B, C, R, Ha, dtype=torch.bool)
                for i, b in enumerate(batch):
                    for c, candidate in enumerate(
                        b.get("ga_rollout_actions", [])
                    ):
                        for r, sequence in enumerate(candidate):
                            for h, action in enumerate(sequence):
                                gra[i, c, r, h, : len(action)] = torch.tensor(
                                    action
                                )
                                gram[i, c, r, h] = True
                extra.update(
                    ga_rollout_action_tokens=gra,
                    ga_rollout_action_mask=gram,
                )
    return {
        **extra,
        "prompt_tokens": prompt_tokens,
        "prompt_mask": prompt_mask,
        "step_tokens": step_tokens,
        "step_mask": step_mask,
        "action_tokens": action_tokens,
        "op": _pad_labels([b["op"] for b in batch]),
        "value": _pad_labels([b["value"] for b in batch]),
        "remaining": _pad_labels([b["remaining"] for b in batch]),
        "resolved_n": _pad_labels([b["resolved_n"] for b in batch]),
        "necessary": _pad_labels([b["necessary"] for b in batch]),
        "answer": torch.tensor([b["answer"] for b in batch]),
        "n_necessary": torch.tensor([b["n_necessary"] for b in batch]),
        "n_vars": torch.tensor([b["n_vars"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch]),
        "var_idx": _pad_labels([b["var_idx"] for b in batch], fill=-1),
        "query_idx": torch.tensor([b["query_idx"] for b in batch]),
        "ancestor_mask": _member_mask([b["ancestors"] for b in batch]),
    }


MAX_VARS = 12


def _member_mask(sets: list[list[int]], width: int = MAX_VARS) -> torch.Tensor:
    out = torch.zeros(len(sets), width, dtype=torch.long)
    for b, s in enumerate(sets):
        for j in s:
            if j < width:
                out[b, j] = 1
    return out
