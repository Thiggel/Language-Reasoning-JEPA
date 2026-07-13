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
        trace = rollout_trace(p, rng, self.distractor_prob, self.max_distractors)

        prompt = [self.vocab.encode(s) for s in prompt_sentences(p, rng)]
        env = SymbolicEnv(p)
        steps, actions, op, value, remaining, resolved_n, necessary = (
            [], [], [], [], [], [], []
        )
        alt_actions: list[list[list[int]]] = []
        alt_remaining: list[list[int]] = []
        for idx in trace:
            actions.append(self.vocab.encode(action_phrase(p, idx)))
            if self.n_alt:
                done = env.resolved_set
                others = [a for a in env.feasible_actions() if a != idx]
                rng.shuffle(others)
                alts = others[: self.n_alt]
                alt_actions.append(
                    [self.vocab.encode(action_phrase(p, a)) for a in alts]
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
        if self.shuffle_actions and len(actions) > 1:
            rng.shuffle(actions)

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
                    "ga_alt_actions": [
                        self.vocab.encode(action_phrase(p, a)) for a in alts
                    ],
                    "ga_alt_steps": [
                        self.vocab.encode(env2.clone().step(a)) for a in alts
                    ],
                }

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
        if self.n_alt:
            out["alt_actions"] = alt_actions
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
    tokens = torch.full((B, T, K, L), pad, dtype=torch.long)
    remaining = torch.full((B, T, K), -1, dtype=torch.long)
    for b, item in enumerate(batch):
        for t, (alts, rems) in enumerate(
            zip(item["alt_actions"], item["alt_remaining"])
        ):
            for k, (a, r) in enumerate(zip(alts, rems)):
                tokens[b, t, k, : len(a)] = torch.tensor(a)
                remaining[b, t, k] = r
    return {"alt_tokens": tokens, "alt_remaining": remaining}


def collate(batch: list[dict], pad_id: int) -> dict:
    prompt_tokens, prompt_mask = _pad_chunks([b["prompt"] for b in batch], pad_id)
    step_tokens, step_mask = _pad_chunks([b["steps"] for b in batch], pad_id)
    action_tokens, _ = _pad_chunks([b["actions"] for b in batch], pad_id)
    extra = _pad_alt(batch, pad_id) if "alt_actions" in batch[0] else {}
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
        for i, b in enumerate(batch):
            if "ga_t" not in b:
                continue
            gat[i] = b["ga_t"]
            for k, (a, st) in enumerate(zip(b["ga_alt_actions"], b["ga_alt_steps"])):
                gaa[i, k, : len(a)] = torch.tensor(a)
                gas[i, k, : len(st)] = torch.tensor(st)
                gav[i, k] = True
        extra.update(ga_t=gat, ga_alt_action_tokens=gaa,
                     ga_alt_step_tokens=gas, ga_valid=gav)
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
