"""Token streams for decoder-only LM baselines.

``LMDataset`` and ``FlattenedDiscourseLMDataset`` train an outcome LM on
prompt + rendered solution steps.  ``IntentPolicyLMDataset`` instead
interleaves intent phrases and observed outcomes and masks the loss to the
intent phrases.  The latter is the information-matched policy baseline: at
test time it ranks the same outcome-free action text as the JEPA planner.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from textjepa.data.igsm.dataset import IGSMDataset
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


class FlattenedDiscourseLMDataset(Dataset):
    """Token-LM view of any dataset that exposes prompt/step token chunks.

    This wrapper is used for official iGSM, where the reference generator and
    vocabulary must remain untouched.  Symbolic preference candidates are
    intentionally not synthesized here; the paper's plain LM baseline only
    receives next-token supervision.
    """

    def __init__(self, discourse: Dataset):
        self.discourse = discourse

    def __len__(self) -> int:
        return len(self.discourse)

    def __getitem__(self, index: int) -> dict:
        item = self.discourse[index]
        prompt = [token for sentence in item["prompt"] for token in sentence]
        steps = [token for sentence in item["steps"] for token in sentence]
        return {"tokens": prompt + steps, "prompt_len": len(prompt)}


class IntentPolicyLMDataset(Dataset):
    """Causal policy view of a discourse trace.

    The stream is ``prompt, intent_1, outcome_1, ..., intent_T, outcome_T``.
    Only intent tokens receive next-token cross-entropy.  Outcomes are still
    appended as observations, making train-time histories identical to those
    constructed by :mod:`scripts.plan_lm` after each selected action.
    """

    def __init__(self, discourse: Dataset):
        self.discourse = discourse

    def __len__(self) -> int:
        return len(self.discourse)

    def __getitem__(self, index: int) -> dict:
        item = self.discourse[index]
        tokens = [token for sentence in item["prompt"] for token in sentence]
        loss_mask = [False] * len(tokens)
        if len(item["actions"]) != len(item["steps"]):
            raise ValueError("every policy action must have one observed outcome")
        for action, outcome in zip(item["actions"], item["steps"]):
            tokens.extend(action)
            loss_mask.extend([True] * len(action))
            tokens.extend(outcome)
            loss_mask.extend([False] * len(outcome))
        return {
            "tokens": tokens,
            "prompt_len": len(tokens),  # unused when loss_mask is present
            "loss_mask": loss_mask,
        }


class IntentSentencePolicyDataset(Dataset):
    """Sentence-level analogue of :class:`IntentPolicyLMDataset`.

    Action and outcome chunks are interleaved in ``steps``.  ``target_mask``
    marks only action chunks, so a sentence LM predicts the next intent while
    retaining past selected intents and observed outcomes as context.
    """

    def __init__(self, discourse: Dataset):
        self.discourse = discourse

    def __len__(self) -> int:
        return len(self.discourse)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.discourse[index])
        if len(item["actions"]) != len(item["steps"]):
            raise ValueError("every policy action must have one observed outcome")
        stream, target_mask = [], []
        for action, outcome in zip(item["actions"], item["steps"]):
            stream.extend([action, outcome])
            target_mask.extend([True, False])
        item["steps"] = stream
        item["target_mask"] = target_mask
        return item


def collate_intent_sentence_policy(batch: list[dict], pad_id: int) -> dict:
    """Use the discourse collator and add the action-chunk target mask."""
    from textjepa.data.igsm.dataset import collate

    out = collate(batch, pad_id)
    B, T = out["step_mask"].shape
    target_mask = torch.zeros((B, T), dtype=torch.bool)
    for i, item in enumerate(batch):
        target_mask[i, : len(item["target_mask"])] = torch.tensor(
            item["target_mask"], dtype=torch.bool
        )
    out["target_mask"] = target_mask
    return out


def collate_lm(batch: list[dict], pad_id: int) -> dict:
    L = max(len(b["tokens"]) for b in batch)
    tokens = torch.full((len(batch), L), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        tokens[i, : len(b["tokens"])] = torch.tensor(b["tokens"])
    out = {
        "tokens": tokens,
        "prompt_len": torch.tensor([b["prompt_len"] for b in batch]),
    }
    if "loss_mask" in batch[0]:
        loss_mask = torch.zeros((len(batch), L), dtype=torch.bool)
        for i, b in enumerate(batch):
            loss_mask[i, : len(b["loss_mask"])] = torch.tensor(
                b["loss_mask"], dtype=torch.bool
            )
        out["loss_mask"] = loss_mask
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
