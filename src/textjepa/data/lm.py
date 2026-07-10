"""Token-stream dataset for the decoder-only LM baseline.

Flattens prompt sentences + step sentences into one token sequence;
loss is masked to the step region (the prompt is context)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.data.vocab import Vocab


class LMDataset(Dataset):
    def __init__(self, vocab: Vocab, size: int, seed: int, **igsm_kwargs):
        igsm_kwargs.pop("shuffle_actions", None)
        igsm_kwargs.pop("n_alt", None)
        self.igsm = IGSMDataset(vocab, size, seed, **igsm_kwargs)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.igsm)

    def __getitem__(self, index: int) -> dict:
        item = self.igsm[index]
        prompt = [t for s in item["prompt"] for t in s]
        steps = [t for s in item["steps"] for t in s]
        return {"tokens": prompt + steps, "prompt_len": len(prompt)}


def collate_lm(batch: list[dict], pad_id: int) -> dict:
    L = max(len(b["tokens"]) for b in batch)
    tokens = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        tokens[i, : len(b["tokens"])] = torch.tensor(b["tokens"])
    return {
        "tokens": tokens,
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
    }
