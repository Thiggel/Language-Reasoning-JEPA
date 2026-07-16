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
        geo_rank_horizon: int = 1,
        geo_rank_rollouts: int = 1,
        geo_rank_policy: str = "random",
        geo_rank_beam_width: int = 1,
        macro_alt_k: int = 0,
        macro_alt_horizon: int = 3,
        all_action_supervision: bool = False,
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
        self.geo_rank_horizon = max(1, int(geo_rank_horizon))
        self.geo_rank_rollouts = max(1, int(geo_rank_rollouts))
        self.geo_rank_policy = str(geo_rank_policy)
        self.geo_rank_beam_width = max(1, int(geo_rank_beam_width))
        self.macro_alt_k = max(0, int(macro_alt_k))
        self.macro_alt_horizon = max(1, int(macro_alt_horizon))
        self.all_action_supervision = bool(all_action_supervision)
        if self.geo_rank_policy not in {"random", "greedy"}:
            raise ValueError(f"unknown geo_rank_policy: {self.geo_rank_policy}")
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
        )
        return p, rng

    def __getitem__(self, index: int) -> dict:
        p, rng = self.problem(index)
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
        if self.geo_rank_k and len(trace) > 1:
            # one anchor step: alt intent phrases + env-rendered TRUE next
            # step sentences (text only; the ranking label is computed in
            # latent space by the model — no symbolic annotations)
            t_star = rng.randrange(len(trace))
            env2 = SymbolicEnv(p)
            for i in trace[:t_star]:
                env2.step(i)
            others = [a for a in env2.feasible_actions() if a != trace[t_star]]
            rng.shuffle(others)
            alts = others[: self.geo_rank_k]
            if alts:
                ga = {
                    "ga_t": t_star,
                    "ga_horizon": self.geo_rank_horizon,
                    "ga_beam_width": self.geo_rank_beam_width,
                    "ga_candidate_ids": [trace[t_star], *alts],
                    "ga_candidate_objects": [trace[t_star], *alts],
                    "ga_alt_actions": [
                        self.vocab.encode(action_phrase(p, a)) for a in alts
                    ],
                    "ga_alt_steps": [
                        self.vocab.encode(env2.clone().step(a)) for a in alts
                    ],
                }
                if self.geo_rank_horizon > 1 and self.geo_rank_policy == "greedy":
                    # The model follows the greedy continuation online because
                    # the policy depends on the current EMA geometry.  Keep the
                    # symbolic problem only as an interaction interface; no
                    # ancestor, remaining-step, or preference labels are used.
                    ga.update(
                        ga_greedy=True,
                        ga_problem=p,
                        ga_trace=list(trace),
                        ga_vocab=self.vocab,
                        ga_env_kind="stylized",
                    )
                elif self.geo_rank_horizon > 1:
                    # Monte-Carlo shooting approximation to an N-step optimal
                    # continuation.  The dataset supplies only feasible action
                    # interactions and rendered text; the model later selects
                    # the rollout with minimum EMA latent goal distance.  No
                    # remaining-step or relevance labels enter that selection.
                    candidates = [trace[t_star], *alts]
                    rollout_steps = []
                    for candidate in candidates:
                        candidate_rollouts = []
                        for _ in range(self.geo_rank_rollouts):
                            roll_env = env2.clone()
                            sequence = list(steps[:t_star])
                            sequence.append(self.vocab.encode(roll_env.step(candidate)))
                            for _depth in range(1, self.geo_rank_horizon):
                                if roll_env.solved:
                                    break
                                feasible = roll_env.feasible_actions()
                                if not feasible:
                                    break
                                nxt = feasible[rng.randrange(len(feasible))]
                                sequence.append(self.vocab.encode(roll_env.step(nxt)))
                            candidate_rollouts.append(sequence)
                        rollout_steps.append(candidate_rollouts)
                    ga["ga_rollout_steps"] = rollout_steps

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
        for b, item in enumerate(batch):
            for v, action in enumerate(item.get("action_candidate_tokens", [])):
                candidate_tokens[b, v, :len(action)] = torch.tensor(action)
                candidate_mask[b, v] = True
            labels = item.get("action_feasible", [])
            if labels:
                tensor = torch.tensor(labels, dtype=torch.bool)
                feasible[b, :tensor.shape[0], :tensor.shape[1]] = tensor
        extra.update(
            action_candidate_tokens=candidate_tokens,
            action_candidate_mask=candidate_mask,
            action_feasible=feasible,
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
                     ga_beam_width=max(
                         (b.get("ga_beam_width", 1) for b in batch)
                     ),
                     ga_candidate_ids=gac,
                     ga_alt_action_tokens=gaa,
                     ga_alt_step_tokens=gas, ga_valid=gav)
        if any(b.get("ga_greedy", False) for b in batch):
            candidate_objects = [
                list(b.get("ga_candidate_objects", [])) for b in batch
            ]
            max_candidates = K + 1
            candidate_objects = [
                row + [None] * (max_candidates - len(row))
                for row in candidate_objects
            ]
            extra.update(
                ga_greedy=True,
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
