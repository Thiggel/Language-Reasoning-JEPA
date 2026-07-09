"""Torch dataset of edit trajectories: corrupted draft -> perfect draft.

Repair edits are interleaved with occasional "vandal" inserts (distractor
steps), so the model sees edits that increase the distance to the goal —
essential negative signal for the value/goal energy.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from textjepa.data.edits.trajectory import EDIT_OP_LABELS, Edit, EditEnv
from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.igsm.render import prompt_sentences
from textjepa.data.vocab import Vocab


class EditDataset(Dataset):
    """Wraps IGSM problem generation; emits edit trajectories."""

    def __init__(
        self,
        vocab: Vocab,
        size: int,
        seed: int,
        max_wrong: int = 2,
        max_missing: int = 1,
        max_extra: int = 1,
        vandal_prob: float = 0.1,
        max_vandal: int = 1,
        **igsm_kwargs,
    ):
        self.igsm = IGSMDataset(vocab, size, seed, **igsm_kwargs)
        self.vocab = vocab
        self.max_wrong = max_wrong
        self.max_missing = max_missing
        self.max_extra = max_extra
        self.vandal_prob = vandal_prob
        self.max_vandal = max_vandal

    def __len__(self) -> int:
        return len(self.igsm)

    def problem(self, index: int):
        return self.igsm.problem(index)

    def make_env(self, index: int) -> tuple[EditEnv, random.Random, list[str]]:
        p, rng = self.igsm.problem(index)
        env = EditEnv(
            p, rng,
            max_wrong=self.max_wrong,
            max_missing=self.max_missing,
            max_extra=self.max_extra,
        )
        return env, rng, prompt_sentences(p, rng)

    def __getitem__(self, index: int) -> dict:
        env, rng, prompt = self.make_env(index)
        p = env.p
        enc = self.vocab.encode

        buffers = [[enc(s) for s in env.sentences()]]
        actions, op, value, remaining, resolved_n, necessary = [], [], [], [], [], []
        n_initial = env.n_defects()
        n_vandal = 0
        while not env.solved:
            fixes = env.fixing_edits()
            harmful: list[Edit] = []
            if n_vandal < self.max_vandal and rng.random() < self.vandal_prob:
                harmful = [
                    e for e in env.candidate_edits(include_harmful=True, rng=rng)
                    if e.kind == "insert" and e.var not in p.query_ancestors
                ]
            vandalize = bool(harmful)
            edit = rng.choice(harmful) if vandalize else rng.choice(fixes)
            n_vandal += int(vandalize)

            actions.append(enc(env.intent_text(edit)))
            env.apply(edit)
            buffers.append([enc(s) for s in env.sentences()])
            op.append(EDIT_OP_LABELS[edit.kind])
            value.append(env.stated_query_value())
            remaining.append(env.n_defects())
            resolved_n.append(len(env.buffer))
            necessary.append(int(not vandalize))

        return {
            "prompt": [enc(s) for s in prompt],
            "buffers": buffers,
            "actions": actions,
            "op": op,
            "value": value,
            "remaining": remaining,
            "resolved_n": resolved_n,
            "necessary": necessary,
            "answer": p.answer,
            "n_necessary": n_initial,
            "n_vars": len(p.vars),
            "index": index,
        }


def collate_edits(batch: list[dict], pad_id: int) -> dict:
    from textjepa.data.igsm.dataset import _pad_chunks, _pad_labels

    prompt_tokens, prompt_mask = _pad_chunks([b["prompt"] for b in batch], pad_id)
    action_tokens, _ = _pad_chunks([b["actions"] for b in batch], pad_id)

    B = len(batch)
    S = max(len(b["buffers"]) for b in batch)  # T+1 buffer snapshots
    C = max((len(buf) for b in batch for buf in b["buffers"]), default=1)
    C = max(C, 1)
    L = max(
        (len(ch) for b in batch for buf in b["buffers"] for ch in buf), default=1
    )
    buffer_tokens = torch.full((B, S, C, L), pad_id, dtype=torch.long)
    buffer_mask = torch.zeros(B, S, C, dtype=torch.bool)
    snapshot_mask = torch.zeros(B, S, dtype=torch.bool)
    for b, item in enumerate(batch):
        for t, buf in enumerate(item["buffers"]):
            snapshot_mask[b, t] = True
            for c, chunk in enumerate(buf):
                buffer_tokens[b, t, c, : len(chunk)] = torch.tensor(chunk)
                buffer_mask[b, t, c] = True

    step_mask = snapshot_mask[:, 1:]
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_mask": prompt_mask,
        "buffer_tokens": buffer_tokens,
        "buffer_mask": buffer_mask,
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
    }
