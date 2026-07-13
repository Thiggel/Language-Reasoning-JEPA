"""Checkpoint loading and dataset construction shared by all scripts."""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from textjepa.data.edits.dataset import EditDataset, collate_edits
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate


def load_run(ckpt_path: str, device: str = "cuda:0", random_init: bool = False):
    """Returns (model, vocab, cfg) from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    if cfg.data.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(cfg.data.modulus)
    model = instantiate(cfg.model, vocab_size=len(vocab), pad_id=vocab.pad_id)
    if not random_init:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
        if missing:
            print(f"note: initializing modules added after this run: {missing}")
    return model.to(device).eval(), vocab, cfg


def build_dataset(cfg, vocab, split: str = "val", size: int | None = None):
    d = cfg.data
    igsm_kwargs = dict(
        modulus=d.modulus,
        n_vars_range=tuple(d.n_vars_range),
        leaf_prob=d.leaf_prob,
        steps_range=tuple(d.steps_range),
        distractor_prob=d.distractor_prob,
        max_distractors=d.max_distractors,
        # .get: configs stored in older checkpoints lack these keys
        shuffle_actions=d.get("shuffle_actions", False),
        n_alt=d.get("n_alt", 0),
        geo_rank_k=d.get("geo_rank_k", 0),
    )
    size = size or (d.val_size if split == "val" else d.train_size)
    seed = d.val_seed if split == "val" else d.train_seed
    if d.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import FaithfulDataset

        return FaithfulDataset(
            vocab, size=size, seed=seed,
            max_op=d.max_op, max_edge=d.max_edge,
            op_range=tuple(d.op_range),
            distractor_prob=d.distractor_prob,
            max_distractors=d.max_distractors,
            n_alt=d.get("n_alt", 0),
        )
    if d.get("name", "igsm") == "igsm_edit":
        kw = dict(igsm_kwargs)
        kw.pop("shuffle_actions", None)
        kw.pop("n_alt", None)
        return EditDataset(
            vocab, size=size, seed=seed,
            max_wrong=d.max_wrong, max_missing=d.max_missing,
            max_extra=d.max_extra, vandal_prob=d.vandal_prob,
            max_vandal=d.max_vandal, n_alt=d.get("n_alt", 0), **kw,
        )
    return IGSMDataset(vocab, size=size, seed=seed, **igsm_kwargs)


def collate_for(cfg):
    """Returns the collate function matching the configured dataset."""
    return collate_edits if cfg.data.get("name", "igsm") == "igsm_edit" else collate
