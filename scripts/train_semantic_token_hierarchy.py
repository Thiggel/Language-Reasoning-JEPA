"""Train the variable-duration phrase/sentence causal JEPA."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from train_token_hierarchy_v2 import compute_losses
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.sampling import FreshEpochSampler
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.semantic_token_hierarchy import SemanticBoundaryTokenHierarchyJEPA
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def make_dataset(cfg, vocab, size, seed):
    return SemanticBoundaryLMDataset(
        vocab, size=size, seed=seed, boundary_mode=cfg.boundary_mode,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


def forward(model, batch, device):
    return model(
        batch["tokens"].to(device), batch["prompt_len"].to(device),
        batch["phrase_ends"].to(device), batch["sentence_ends"].to(device),
    )


@hydra.main(config_path="../configs", config_name="semantic_token_hierarchy", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.modulus)
    train_ds = make_dataset(cfg, vocab, cfg.data.train_size, cfg.data.train_seed)
    val_ds = make_dataset(cfg, vocab, cfg.data.val_size, cfg.data.val_seed)
    sampler = FreshEpochSampler(train_ds, seed=cfg.seed)
    collate = partial(collate_semantic_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers, collate_fn=collate,
    )
    model = SemanticBoundaryTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(cfg.device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    total_steps = cfg.train.epochs * len(train_loader)
    logger = MetricLogger(out_dir)
    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = cfg.train.lr * cosine_warmup(
                    step, total_steps, cfg.train.warmup_steps
                )
            out = forward(model, batch, cfg.device)
            loss, items = compute_losses(out, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            model.update_teacher(ema_momentum(
                step, total_steps, cfg.train.ema_start, cfg.train.ema_end
            ))
            if step % cfg.train.log_every == 0:
                logger.log(step, {
                    name: float(value.detach()) for name, value in items.items()
                }, prefix="train/")
            step += 1
        model.eval()
        sums, count, state_features, level_features = {}, 0, [], None
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= cfg.train.eval_batches:
                    break
                out = forward(model, batch, cfg.device)
                _, items = compute_losses(out, cfg)
                for name, value in items.items():
                    sums[name] = sums.get(name, 0.0) + float(value)
                state_features.append(out["prev"][out["valid"]])
                if level_features is None:
                    level_features = [[] for _ in out["levels"]]
                for level_index, level in enumerate(out["levels"]):
                    level_features[level_index].append(level["codes"][level["valid"]])
                count += 1
        metrics = {name: value / count for name, value in sums.items()}
        states = torch.cat(state_features)
        metrics.update(
            state_std=feature_std(states),
            state_rank=effective_rank(states[:4096]),
        )
        for index, features in enumerate(level_features or []):
            codes = torch.cat(features)
            metrics[f"level{index + 1}_action_std"] = feature_std(codes)
            metrics[f"level{index + 1}_action_rank"] = effective_rank(codes[:4096])
        logger.log(step, metrics, prefix="val/")
        payload = {
            "model": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch, "metrics": metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if metrics["selection"] < best:
            best = metrics["selection"]
            torch.save(payload, out_dir / "best.pt")
        print(f"[epoch {epoch}] " + "  ".join(
            f"{name}={value:.4f}" for name, value in sorted(metrics.items())
        ), flush=True)
    logger.close()


if __name__ == "__main__":
    main()
