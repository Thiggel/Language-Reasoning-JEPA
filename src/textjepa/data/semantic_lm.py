"""Token streams with phrase and sentence boundary annotations."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from textjepa.data.igsm.dataset import IGSMDataset


def semantic_phrase_ends(step: list[int], id_to_token: list[str]) -> list[int]:
    """Return exclusive phrase ends for one rendered reasoning sentence.

    The controlled iGSM renderer has explicit copular and equality markers.
    Cutting immediately before ``is`` and ``=`` separates the entity phrase,
    computation phrase, and result phrase without consulting symbolic
    feasibility or the underlying graph.
    """
    cuts = [
        index for index, token_id in enumerate(step)
        if index > 0 and id_to_token[int(token_id)] in {"is", "="}
    ]
    return sorted(set(cuts + [len(step)]))


def random_matched_phrase_ends(
    step: list[int], count: int, rng: random.Random
) -> list[int]:
    """Random boundaries with the same number of segments as semantics."""
    internal = max(0, count - 1)
    candidates = list(range(1, len(step)))
    chosen = sorted(rng.sample(candidates, min(internal, len(candidates))))
    return chosen + [len(step)]


class SemanticBoundaryLMDataset(Dataset):
    def __init__(self, vocab, size: int, seed: int, boundary_mode="semantic", **kwargs):
        if boundary_mode not in {"semantic", "random_matched"}:
            raise ValueError(f"unknown boundary mode: {boundary_mode}")
        self.igsm = IGSMDataset(vocab, size, seed, **kwargs)
        self.vocab = vocab
        self.seed = seed
        self.boundary_mode = boundary_mode

    def __len__(self):
        return len(self.igsm)

    def __getitem__(self, index):
        item = self.igsm[index]
        prompt = [token for sentence in item["prompt"] for token in sentence]
        reasoning, phrase_ends, sentence_ends = [], [], []
        offset = 0
        rng = random.Random(f"{self.seed}:{index}:phrase-boundaries")
        for step in item["steps"]:
            semantic = semantic_phrase_ends(step, self.vocab.id_to_token)
            local = (
                semantic if self.boundary_mode == "semantic"
                else random_matched_phrase_ends(step, len(semantic), rng)
            )
            phrase_ends.extend(offset + end for end in local)
            offset += len(step)
            sentence_ends.append(offset)
            reasoning.extend(step)
        return {
            "tokens": prompt + reasoning,
            "prompt_len": len(prompt),
            "phrase_ends": phrase_ends,
            "sentence_ends": sentence_ends,
        }


def collate_semantic_lm(batch, pad_id: int):
    batch_size = len(batch)
    width = max(len(item["tokens"]) for item in batch)
    phrase_width = max(len(item["phrase_ends"]) for item in batch)
    sentence_width = max(len(item["sentence_ends"]) for item in batch)
    tokens = torch.full((batch_size, width), pad_id, dtype=torch.long)
    phrases = torch.full((batch_size, phrase_width), -1, dtype=torch.long)
    sentences = torch.full((batch_size, sentence_width), -1, dtype=torch.long)
    for row, item in enumerate(batch):
        tokens[row, :len(item["tokens"])] = torch.tensor(item["tokens"])
        phrases[row, :len(item["phrase_ends"])] = torch.tensor(item["phrase_ends"])
        sentences[row, :len(item["sentence_ends"])] = torch.tensor(item["sentence_ends"])
    return {
        "tokens": tokens,
        "prompt_len": torch.tensor([item["prompt_len"] for item in batch]),
        "phrase_ends": phrases,
        "sentence_ends": sentences,
    }
