"""Train a TextJEPA model. Usage:

    python scripts/train.py run_name=my_run model.d_action=8 train.epochs=30
"""

from __future__ import annotations

from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
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
    out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(OmegaConf.to_yaml(cfg))

    vocab = build_vocab(cfg.data.modulus)
    train_ds = build_dataset(cfg, vocab, split="train")
    val_ds = build_dataset(cfg, vocab, split="val")
    coll = partial(collate_for(cfg), pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {n_params / 1e6:.2f}M")

    trainer = Trainer(cfg, model, build_objective(cfg), train_loader, val_loader, out_dir)
    trainer.fit()


if __name__ == "__main__":
    main()
