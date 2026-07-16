"""Token-edit trajectories over official iGSM solution text.

Corruption and repair operate only on rendered tokens.  No dependency graph,
remaining-work label, feasibility signal, or symbolic action ordering enters
the edit trajectory.
"""

from __future__ import annotations

import random

from torch.utils.data import Dataset

from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab
from textjepa.data.vocab import Vocab


TOKEN_EDIT_WORDS = ["token", "position", "with"]
OPS = {"delete": 0, "insert": 1, "replace": 2}


def faithful_token_edit_vocab(max_position: int = 1024) -> Vocab:
    base = cached_faithful_vocab()
    words = [
        token for token in base.token_to_id
        if token not in {base.PAD, base.UNK}
    ]
    words.extend(TOKEN_EDIT_WORDS)
    words.extend(str(index) for index in range(max_position + 1))
    return Vocab(words)


def _sentences(tokens: list[int], period_id: int) -> list[list[int]]:
    chunks, start = [], 0
    for index, token in enumerate(tokens):
        if token == period_id:
            chunks.append(tokens[start:index + 1])
            start = index + 1
    if start < len(tokens):
        chunks.append(tokens[start:])
    return chunks or [[]]


def _apply(tokens: list[int], action: tuple[str, int, int | None]) -> None:
    kind, position, token = action
    if kind == "delete":
        tokens.pop(position)
    elif kind == "insert":
        tokens.insert(position, int(token))
    elif kind == "replace":
        tokens[position] = int(token)
    else:
        raise ValueError(kind)


class FaithfulTokenEditDataset(Dataset):
    """Official iGSM prompt plus a generically corrupted solution buffer."""

    def __init__(self, vocab: Vocab, size: int, seed: int, max_op: int = 21,
                 max_edge: int = 28, op_range=(8, 21), min_edits: int = 6,
                 max_edits: int = 16, **_):
        self.vocab = vocab
        self.seed = seed
        self.min_edits = int(min_edits)
        self.max_edits = int(max_edits)
        self.source = FaithfulDataset(
            vocab, size=size, seed=seed, max_op=max_op, max_edge=max_edge,
            op_range=tuple(op_range), distractor_prob=0.0,
            max_distractors=0,
        )
        self.period_id = vocab.token_to_id["."]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index: int) -> dict:
        source = self.source[index]
        prompt = source["prompt"]
        target = [token for sentence in source["steps"] for token in sentence]
        rng = random.Random(f"faithful-token-edit:{self.seed}:{index}")
        current = list(target)
        undo: list[tuple[str, int, int | None]] = []
        upper = min(self.max_edits, max(self.min_edits, len(target) // 8))
        count = rng.randint(min(self.min_edits, upper), upper)
        token_pool = list(dict.fromkeys(target))
        for _ in range(count):
            allowed = ["replace", "insert"]
            if len(current) > 2:
                allowed.append("delete")
            kind = rng.choice(allowed)
            if kind == "replace":
                position = rng.randrange(len(current))
                old = current[position]
                alternatives = [token for token in token_pool if token != old]
                if not alternatives:
                    continue
                current[position] = rng.choice(alternatives)
                undo.append(("replace", position, old))
            elif kind == "delete":
                position = rng.randrange(len(current))
                old = current.pop(position)
                undo.append(("insert", position, old))
            else:
                position = rng.randrange(len(current) + 1)
                current.insert(position, rng.choice(token_pool))
                undo.append(("delete", position, None))

        repairs = list(reversed(undo))
        buffers = [_sentences(current, self.period_id)]
        actions, op, remaining, changed = [], [], [], []
        for step, action in enumerate(repairs):
            kind, position, token = action
            rendered = f"{kind} token position {position}"
            if token is not None:
                rendered += f" with {self.vocab.id_to_token[int(token)]}"
            rendered += " ."
            actions.append(self.vocab.encode(rendered))
            op.append(OPS[kind])
            _apply(current, action)
            buffers.append(_sentences(current, self.period_id))
            remaining.append(len(repairs) - step - 1)
            changed.append([])
        if current != target:
            raise AssertionError("token-edit undo trajectory did not recover target")
        return {
            "prompt": prompt,
            "buffers": buffers,
            "actions": actions,
            "op": op,
            "value": [0] * len(actions),
            "remaining": remaining,
            "resolved_n": [sum(len(s) for s in b) for b in buffers[1:]],
            "necessary": [1] * len(actions),
            "answer": int(source["answer"]),
            "n_necessary": len(actions),
            "n_vars": 0,
            "index": index,
            "edit_pos": [min(action[1], 15) for action in repairs],
            "changed": changed,
            "defect_masks": [[] for _ in repairs],
            "target_tokens": target,
        }


__all__ = [
    "FaithfulTokenEditDataset", "collate_edits", "faithful_token_edit_vocab",
]
