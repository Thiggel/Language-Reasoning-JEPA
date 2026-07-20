"""Train a TextJEPA model. Usage:

    python scripts/train.py run_name=my_run model.d_action=8 train.epochs=30
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.sampling import FreshEpochSampler, GroupedTrajectoryBatchSampler
from textjepa.objectives import CompositeObjective
from textjepa.training import Trainer
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for


def build_objective(cfg: DictConfig) -> CompositeObjective:
    objectives, weights = {}, {}
    for name, node in cfg.objective.items():
        d = OmegaConf.to_container(node, resolve=True)
        weights[name] = d.pop("weight", 1.0)
        objectives[name] = hydra.utils.instantiate(d)
    return CompositeObjective(objectives, weights)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    predictor_kind = cfg.model.get("predictor_kind")
    if (
        predictor_kind is not None
        and predictor_kind != "causal"
        and not cfg.get("allow_legacy_predictor", False)
    ):
        raise ValueError(
            "new experiments require model.predictor_kind=causal; set "
            "allow_legacy_predictor=true only for an explicit historical audit"
        )
    if (
        int(cfg.model.get("macro_k", 0)) > 0
        and cfg.model.get("high_predictor_kind", "causal") != "causal"
        and not cfg.get("allow_legacy_predictor", False)
    ):
        raise ValueError(
            "new high-level experiments require a causal transformer predictor"
        )
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))

    if cfg.data.get("name", "igsm") == "igsm_real_token_edit":
        from textjepa.data.faithful_token_edits import faithful_token_edit_vocab

        vocab = faithful_token_edit_vocab()
    elif cfg.data.get("name", "igsm") == "igsm_real":
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(cfg.data.modulus)
    train_ds = build_dataset(cfg, vocab, split="train")
    val_ds = build_dataset(cfg, vocab, split="val")
    coll = partial(collate_for(cfg), pad_id=vocab.pad_id)
    fresh_sampler = (
        FreshEpochSampler(train_ds, seed=cfg.seed)
        if cfg.data.get("fresh_per_epoch", True) else None
    )
    trajectory_variants = int(cfg.data.get("trajectory_variants", 1))
    effective_batch_size = int(cfg.train.batch_size)
    microbatch_size = int(cfg.train.get("microbatch_size", effective_batch_size))
    if effective_batch_size % microbatch_size:
        raise ValueError("microbatch_size must divide train.batch_size")
    if trajectory_variants > 1:
        if effective_batch_size % trajectory_variants:
            raise ValueError("batch_size must be divisible by trajectory_variants")
        grouped = GroupedTrajectoryBatchSampler(
            base_size=int(cfg.data.train_size),
            variants=trajectory_variants,
            bases_per_batch=effective_batch_size // trajectory_variants,
            seed=cfg.seed,
            fresh_per_epoch=cfg.data.get("fresh_per_epoch", True),
            microbatch_size=microbatch_size,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=grouped,
            num_workers=cfg.train.num_workers, collate_fn=coll,
            persistent_workers=cfg.train.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=microbatch_size,
            shuffle=fresh_sampler is None,
            sampler=fresh_sampler,
            num_workers=cfg.train.num_workers,
            collate_fn=coll,
            drop_last=True,
            persistent_workers=cfg.train.num_workers > 0,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=coll,
    )

    model = hydra.utils.instantiate(
        cfg.model, vocab_size=len(vocab), pad_id=vocab.pad_id
    )
    init_ckpt = cfg.train.get("init_ckpt")
    if init_ckpt:
        payload = torch.load(
            Path(init_ckpt), map_location="cpu", weights_only=False
        )
        current = model.state_dict()
        # Hierarchy sweeps normally reuse only the low-level base.  Focused
        # head/dynamics controls can instead preserve the complete checkpoint
        # and explicitly reset only the component under study.
        high_prefixes = (
            "core.macro_encoder.",
            "core.hi_predictor.",
            "core.hi_value_head.",
            "core.macro_value_head.",
            "core.macro_support_head.",
            "core.action_support_head.",
            "core.subgoal_action_head.",
            "core.controller_remaining_head.",
            "core.controller_residual_head.",
        )
        preserve_high = cfg.train.get("init_mode", "low_level") == "full"
        compatible = {
            name: value
            for name, value in payload["model"].items()
            if name in current
            and current[name].shape == value.shape
            and (preserve_high or not name.startswith(high_prefixes))
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"initialized {len(compatible)} low-level tensors from {init_ckpt}; "
            f"left {len(missing)} tensors trainable/reinitialized, "
            f"ignored {len(unexpected)}",
            flush=True,
        )
        reset_names = []
        if cfg.train.get("reset_hi_predictor", False):
            reset_names.append("hi_predictor")
        if cfg.train.get("reset_macro_value_head", False):
            reset_names.append("macro_value_head")
        for name in reset_names:
            module = getattr(model.core, name)
            for child in module.modules():
                reset = getattr(child, "reset_parameters", None)
                if reset is not None:
                    reset()
            print(f"reinitialized core.{name}", flush=True)
    if cfg.train.get("freeze_low_level", False):
        model.requires_grad_(False)
        if cfg.train.get("train_high_level", True):
            model.core.macro_encoder.requires_grad_(True)
            model.core.hi_predictor.requires_grad_(True)
            model.core.hi_value_head.requires_grad_(True)
            model.core.macro_value_head.requires_grad_(True)
            model.core.macro_support_head.requires_grad_(True)
            model.core.subgoal_action_head.requires_grad_(True)
        if cfg.train.get("train_hi_predictor", False):
            model.core.hi_predictor.requires_grad_(True)
        if cfg.train.get("train_hi_value_head", False):
            model.core.hi_value_head.requires_grad_(True)
        if cfg.train.get("train_subgoal_action_head", False):
            model.core.subgoal_action_head.requires_grad_(True)
        if cfg.train.get("train_macro_value_head", False):
            model.core.macro_value_head.requires_grad_(True)
        if cfg.train.get("train_action_support", False):
            model.core.action_support_head.requires_grad_(True)
        if cfg.train.get("train_low_predictor", False):
            model.core.predictor.requires_grad_(True)
        if cfg.train.get("train_low_value_head", False):
            model.core.value_head.requires_grad_(True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {n_params / 1e6:.2f}M")

    trainer = Trainer(cfg, model, build_objective(cfg), train_loader, val_loader, out_dir)
    trainer.fit()


if __name__ == "__main__":
    main()
