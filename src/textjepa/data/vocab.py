"""Whitespace word-level vocabulary for synthetic corpora.

All renderers emit space-separated tokens, so tokenization is a split.
"""

from __future__ import annotations

# Words used by the edit-track intent phrases; included in every vocab so
# both tracks share one tokenizer (defined here to avoid import cycles).
EDIT_WORDS = [
    "delete", "insert", "replace", "recompute", "step", "after", "at",
    "start", ":",
]


class Vocab:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, tokens: list[str]):
        specials = [self.PAD, self.UNK]
        seen: dict[str, int] = {}
        for tok in specials + tokens:
            if tok not in seen:
                seen[tok] = len(seen)
        self.token_to_id = seen
        self.id_to_token = {i: t for t, i in seen.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id[self.UNK]
        return [self.token_to_id.get(t, unk) for t in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.id_to_token[i] for i in ids if i != self.pad_id)
