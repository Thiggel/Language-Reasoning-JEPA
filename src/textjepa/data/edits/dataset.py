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
        n_alt: int = 0,
        **igsm_kwargs,
    ):
        igsm_kwargs.pop("shuffle_actions", None)  # discourse-only control
        igsm_kwargs.pop("n_alt", None)
        self.igsm = IGSMDataset(vocab, size, seed, **igsm_kwargs)
        self.vocab = vocab
        self.max_wrong = max_wrong
        self.max_missing = max_missing
        self.max_extra = max_extra
        self.vandal_prob = vandal_prob
        self.max_vandal = max_vandal
        self.n_alt = n_alt  # counterfactual candidate edits per step

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
        alt_actions: list[list[list[int]]] = []
        alt_remaining: list[list[int]] = []
        edit_pos: list[int] = []
        changed: list[list[int]] = []
        defect_masks: list[list[int]] = []
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

            if self.n_alt:
                others = [
                    e for e in env.candidate_edits(include_harmful=True, rng=rng)
                    if e != edit
                ]
                rng.shuffle(others)
                alts = others[: self.n_alt]
                alt_actions.append([enc(env.intent_text(e)) for e in alts])
                rems = []
                for e in alts:
                    c = env.clone()
                    c.apply(e)
                    rems.append(c.n_defects())
                alt_remaining.append(rems)

            actions.append(enc(env.intent_text(edit)))
            env.apply(edit)
            buffers.append([enc(s) for s in env.sentences()])
            changed.append(
                enc(env.buffer[edit.pos].text)
                if edit.kind != "delete" and edit.pos < len(env.buffer)
                else []
            )
            edit_pos.append(min(edit.pos, 15))
            defect_masks.append(
                [int(env.is_defect(b)) for b in env.buffer[:16]]
            )
            op.append(EDIT_OP_LABELS[edit.kind])
            value.append(env.stated_query_value())
            remaining.append(env.n_defects())
            resolved_n.append(len(env.buffer))
            necessary.append(int(not vandalize))

        out = {
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
            "edit_pos": edit_pos,
            "changed": changed,
            "defect_masks": defect_masks,
        }
        if self.n_alt:
            out["alt_actions"] = alt_actions
            out["alt_remaining"] = alt_remaining
        return out


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
    from textjepa.data.igsm.dataset import _pad_alt

    if "alt_buffers" in batch[0]:
        extra = _pad_alt_buffers(batch, pad_id)
    else:
        extra = _pad_alt(batch, pad_id) if "alt_actions" in batch[0] else {}
    return {
        **extra,
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
        "edit_pos": _pad_labels([b["edit_pos"] for b in batch], fill=-1),
        "changed_tokens": _pad_chunks(
            [[c or [pad_id] for c in b["changed"]] for b in batch], pad_id
        )[0],
        "changed_valid": torch.tensor(
            [[bool(c) for c in b["changed"]]
             + [False] * (max(len(x["changed"]) for x in batch) - len(b["changed"]))
             for b in batch]
        ),
        "defect_mask": _pad_defects([b["defect_masks"] for b in batch]),
    }


def _pad_alt_buffers(batch: list[dict], pad: int) -> dict:
    """Pad unlabeled edit outcomes without manufacturing quality labels."""
    B = len(batch)
    T = max(len(item["alt_actions"]) for item in batch)
    K = max(
        (len(step) for item in batch for step in item["alt_actions"]),
        default=1,
    )
    La = max(
        (len(action) for item in batch for step in item["alt_actions"]
         for action in step),
        default=1,
    )
    C = max(
        (len(outcome) for item in batch for step in item["alt_buffers"]
         for outcome in step),
        default=1,
    )
    L = max(
        (len(sentence) for item in batch for step in item["alt_buffers"]
         for outcome in step for sentence in outcome),
        default=1,
    )
    tokens = torch.full((B, T, K, La), pad, dtype=torch.long)
    outcomes = torch.full((B, T, K, C, L), pad, dtype=torch.long)
    outcome_mask = torch.zeros((B, T, K, C), dtype=torch.bool)
    valid = torch.zeros((B, T, K), dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        for step_index, (actions, buffers) in enumerate(zip(
            item["alt_actions"], item["alt_buffers"]
        )):
            for candidate, (action, buffer) in enumerate(zip(actions, buffers)):
                valid[batch_index, step_index, candidate] = True
                tokens[batch_index, step_index, candidate, :len(action)] = (
                    torch.tensor(action)
                )
                for sentence_index, sentence in enumerate(buffer):
                    outcomes[
                        batch_index, step_index, candidate, sentence_index,
                        :len(sentence)
                    ] = torch.tensor(sentence)
                    outcome_mask[
                        batch_index, step_index, candidate, sentence_index
                    ] = True
    return {
        "alt_tokens": tokens,
        "alt_buffer_tokens": outcomes,
        "alt_buffer_mask": outcome_mask,
        "alt_valid": valid,
    }


def _pad_defects(masks: list[list[list[int]]], width: int = 16) -> torch.Tensor:
    """[B, T, 16] per-position defect flags; -1 marks absent positions."""
    B = len(masks)
    T = max(len(m) for m in masks)
    out = torch.full((B, T, width), -1, dtype=torch.long)
    for b, steps in enumerate(masks):
        for t, flags in enumerate(steps):
            out[b, t, : len(flags)] = torch.tensor(flags)
    return out
