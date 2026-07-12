"""Token-stream dataset for the decoder-only LM baseline.

Flattens prompt sentences + step sentences into one token sequence;
loss is masked to the step region (the prompt is context)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.data.vocab import Vocab


class LMDataset(Dataset):
    def __init__(self, vocab: Vocab, size: int, seed: int, n_alt: int = 0,
                 **igsm_kwargs):
        igsm_kwargs.pop("shuffle_actions", None)
        igsm_kwargs.pop("n_alt", None)
        self.igsm = IGSMDataset(vocab, size, seed, **igsm_kwargs)
        self.vocab = vocab
        self.n_alt = n_alt  # DPO-style ranking candidates per trace

    def __len__(self) -> int:
        return len(self.igsm)

    def __getitem__(self, index: int) -> dict:
        item = self.igsm[index]
        prompt = [t for s in item["prompt"] for t in s]
        steps = [t for s in item["steps"] for t in s]
        out = {"tokens": prompt + steps, "prompt_len": len(prompt)}
        if self.n_alt:
            out.update(self._rank_sample(index, item, prompt))
        return out

    def _rank_sample(self, index: int, item: dict, prompt: list) -> dict:
        """One ranking anchor per trace: prefix + executed step sentence vs
        K alternative feasible steps' sentences, with better/worse labels
        (same alt-outcome supervision as the JEPA ranking loss)."""
        import random

        from textjepa.data.igsm.render import step_sentence

        p, _ = self.igsm.problem(index)
        rng = random.Random(f"{self.igsm.seed}:{index}:rank")
        env = SymbolicEnv(p)
        # replay the trace to a random anchor step
        n_steps = len(item["steps"])
        t_star = rng.randrange(n_steps)
        prefix = list(prompt)
        executed = None
        for t in range(t_star + 1):
            feas = env.feasible_actions()
            # recover the executed var by matching the step tokens
            target = item["steps"][t]
            executed = None
            for a in feas:
                if self.vocab.encode(step_sentence(p, a)) == target:
                    executed = a
                    break
            if executed is None:
                executed = feas[0]
            if t < t_star:
                prefix += item["steps"][t]
                env.step(executed)
        done = env.resolved_set
        feas = env.feasible_actions()
        alts = [a for a in feas if a != executed]
        rng.shuffle(alts)
        alts = alts[: self.n_alt]
        rem_exec = len(p.query_ancestors - (done | {executed}))
        cands = [item["steps"][t_star]] + [
            self.vocab.encode(step_sentence(p, a)) for a in alts
        ]
        better = []
        for a in alts:
            rem_a = len(p.query_ancestors - (done | {a}))
            better.append(1 if rem_exec < rem_a else (-1 if rem_exec > rem_a else 0))
        return {"rank_prefix": prefix, "rank_cands": cands, "rank_better": better}


def collate_lm(batch: list[dict], pad_id: int) -> dict:
    L = max(len(b["tokens"]) for b in batch)
    tokens = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        tokens[i, : len(b["tokens"])] = torch.tensor(b["tokens"])
    out = {
        "tokens": tokens,
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
    }
    if "rank_prefix" in batch[0]:
        K1 = max(len(b["rank_cands"]) for b in batch)
        L2 = max(
            len(b["rank_prefix"]) + max(len(c) for c in b["rank_cands"])
            for b in batch
        )
        rt = torch.full((len(batch), K1, L2), pad_id, dtype=torch.long)
        rb = torch.zeros(len(batch), K1 - 1, dtype=torch.long)
        rf = torch.tensor([len(b["rank_prefix"]) for b in batch])
        for i, b in enumerate(batch):
            for k, c in enumerate(b["rank_cands"]):
                seq = b["rank_prefix"] + c
                rt[i, k, : len(seq)] = torch.tensor(seq)
            for k, v in enumerate(b["rank_better"]):
                rb[i, k] = v
        out.update(rank_tokens=rt, rank_better=rb, rank_from=rf)
    return out
