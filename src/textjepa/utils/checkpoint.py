"""Checkpoint loading and dataset construction shared by all scripts."""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from textjepa.data.edits.dataset import EditDataset, collate_edits
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate


def _migrate_legacy_state_dict(state: dict[str, torch.Tensor]):
    """Adapt checkpoints created before ``MacroActionModel`` wrapped its encoder.

    Historical transformer macro encoders lived directly at
    ``core.macro_encoder.*``.  The current module keeps the identical encoder
    under ``core.macro_encoder.encoder.*`` and adds a conditional prior.  Only
    the historical encoder tensors are renamed; the newly added prior remains
    randomly initialized and is reported through the ordinary missing-key
    path.
    """
    legacy_marker = "core.macro_encoder.cls"
    current_marker = "core.macro_encoder.encoder.cls"
    if legacy_marker not in state or current_marker in state:
        return state
    prefix = "core.macro_encoder."
    migrated = {}
    for name, value in state.items():
        if name.startswith(prefix):
            name = prefix + "encoder." + name[len(prefix):]
        migrated[name] = value
    return migrated


def load_run(ckpt_path: str, device: str = "cuda:0", random_init: bool = False):
    """Returns (model, vocab, cfg) from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    if cfg.data.get("name", "igsm") == "igsm_real_token_edit":
        from textjepa.data.faithful_token_edits import faithful_token_edit_vocab

        vocab = faithful_token_edit_vocab()
    elif cfg.data.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(cfg.data.modulus)
    model = instantiate(cfg.model, vocab_size=len(vocab), pad_id=vocab.pad_id)
    if not random_init:
        state = _migrate_legacy_state_dict(ckpt["model"])
        missing, unexpected = model.load_state_dict(state, strict=False)
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
        geo_rank_horizon=d.get("geo_rank_horizon", 1),
        geo_rank_rollouts=d.get("geo_rank_rollouts", 1),
        geo_rank_policy=d.get("geo_rank_policy", "random"),
        geo_rank_beam_width=d.get("geo_rank_beam_width", 1),
        macro_alt_k=d.get("macro_alt_k", 0),
        macro_alt_horizon=d.get("macro_alt_horizon", 3),
        all_action_supervision=d.get("all_action_supervision", False),
    )
    if split == "train":
        default_size, seed = d.train_size, d.train_seed
    elif split == "val":
        default_size, seed = d.val_size, d.val_seed
    elif split == "test":
        default_size = d.get("test_size", d.val_size)
        seed = d.get("test_seed", int(d.val_seed) + 1)
    else:
        raise ValueError(f"unknown dataset split: {split}")
    size = size or default_size
    if d.get("name", "igsm") == "igsm_real_token_edit":
        from textjepa.data.faithful_token_edits import FaithfulTokenEditDataset

        return FaithfulTokenEditDataset(
            vocab, size=size, seed=seed, max_op=d.max_op,
            max_edge=d.max_edge, op_range=tuple(d.op_range),
            min_edits=d.min_edits, max_edits=d.max_edits,
        )
    if d.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import FaithfulDataset

        return FaithfulDataset(
            vocab, size=size, seed=seed,
            max_op=d.max_op, max_edge=d.max_edge,
            op_range=tuple(d.op_range),
            distractor_prob=d.distractor_prob,
            max_distractors=d.max_distractors,
            n_alt=d.get("n_alt", 0),
            geo_rank_k=d.get("geo_rank_k", 0),
            geo_rank_horizon=d.get("geo_rank_horizon", 1),
            geo_rank_rollouts=d.get("geo_rank_rollouts", 1),
            geo_rank_policy=d.get("geo_rank_policy", "random"),
            geo_rank_beam_width=d.get("geo_rank_beam_width", 1),
            macro_alt_k=d.get("macro_alt_k", 0),
            macro_alt_horizon=d.get("macro_alt_horizon", 3),
            all_action_supervision=d.get("all_action_supervision", False),
        )
    if d.get("name", "igsm") == "igsm_edit":
        kw = dict(igsm_kwargs)
        kw.pop("shuffle_actions", None)
        kw.pop("n_alt", None)
        kw.pop("macro_alt_k", None)
        kw.pop("macro_alt_horizon", None)
        kw.pop("all_action_supervision", None)
        return EditDataset(
            vocab, size=size, seed=seed,
            max_wrong=d.max_wrong, max_missing=d.max_missing,
            max_extra=d.max_extra, vandal_prob=d.vandal_prob,
            max_vandal=d.max_vandal, n_alt=d.get("n_alt", 0), **kw,
        )
    return IGSMDataset(vocab, size=size, seed=seed, **igsm_kwargs)


def collate_for(cfg):
    """Returns the collate function matching the configured dataset."""
    return (
        collate_edits
        if cfg.data.get("name", "igsm") in {"igsm_edit", "igsm_real_token_edit"}
        else collate
    )
