"""Measure complete 124M-scale training steps at one microbatch size.

This intentionally includes data collation, the real objective (including GAR
for JEPA), backward, clipping, and fused AdamW.  It is a throughput gate, not a
scientific experiment.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    from train_lm import lm_loss
    from train_pooled_sentence_jepa import compute_losses, forward as jepa_forward
except ModuleNotFoundError:
    from scripts.train_lm import lm_loss
    from scripts.train_pooled_sentence_jepa import compute_losses, forward as jepa_forward
from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.pooled_sentence_jepa import PooledSentenceJEPA
from textjepa.models.sent_lm import SentenceLM
from textjepa.training.optim import build_optimizer
from textjepa.training.trainer import to_device


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=(
        "token_lm", "sentence_lm", "jepa_prior", "jepa_no_prior",
        "jepa_visreg", "jepa_visreg_d2_nogar",
        "jepa_visreg_d1_gar", "jepa_visreg_d1_nogar",
        "jepa_visreg_packed", "jepa_visreg_packed_d1_gar",
    ))
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def data_kwargs():
    return dict(
        modulus=23, n_vars_range=(10, 18), leaf_prob=0.35,
        steps_range=(6, 12), distractor_prob=0.15, max_distractors=2,
    )


def build(kind, batch_size, device):
    vocab = build_vocab(23)
    if kind == "token_lm":
        dataset = LMDataset(vocab, size=batch_size, seed=173, n_alt=0, **data_kwargs())
        batch = next(iter(DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=partial(collate_lm, pad_id=vocab.pad_id),
        )))
        model = DecoderLM(
            len(vocab), vocab.pad_id, d_model=888, n_layers=13,
            n_heads=12, ff_mult=4, max_len=768,
        ).to(device)
        loss_fn = lambda module: lm_loss(module, batch, device)
    elif kind == "sentence_lm":
        dataset = IGSMDataset(vocab, size=batch_size, seed=173, **data_kwargs())
        batch = next(iter(DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=partial(collate, pad_id=vocab.pad_id),
        )))
        model = SentenceLM(
            len(vocab), vocab.pad_id, d_model=768, chunk_layers=2,
            chunk_heads=12, state_layers=11, state_heads=12, dec_layers=3,
            dec_heads=12, ff_mult=4, max_chunk_len=96, max_chunks=64,
            latent_target=False,
        ).to(device)
        gpu_batch = to_device(batch, device)
        loss_fn = lambda module: sum(module(gpu_batch).values())
    else:
        dataset = SemanticBoundaryLMDataset(
            vocab, size=batch_size, seed=173, boundary_mode="semantic",
            **data_kwargs(),
        )
        batch = next(iter(DataLoader(
            dataset, batch_size=batch_size,
            collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
        )))
        model = PooledSentenceJEPA(
            len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
            question_id=vocab.token_to_id["?"], d_state=768,
            encoder_layers=10, pool_heads=12, predictor_layers=6,
            n_heads=12, ff_mult=4, max_len=768, d_action=128,
            dense_depth=(
                1 if kind in {
                    "jepa_visreg_d1_gar", "jepa_visreg_d1_nogar",
                    "jepa_visreg_packed_d1_gar",
                }
                else 2
            ),
            dense_checkpoint=True, pooling_scope="sentence",
            use_prefix_decoder=False,
            use_token_prior=kind in {
                "jepa_prior", "jepa_visreg", "jepa_visreg_d2_nogar",
                "jepa_visreg_d1_gar", "jepa_visreg_d1_nogar",
                "jepa_visreg_packed",
            },
            target_mode=("visreg" if kind.startswith("jepa_visreg") else "ema"),
            visreg_projections=4096,
            attention_backend="auto",
            sequence_packing=kind in {
                "jepa_visreg_packed", "jepa_visreg_packed_d1_gar",
            },
        ).to(device)
        cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/pooled_sentence_jepa.yaml")
        cfg.objective.dense_discount = 1.0
        cfg.objective.token_prior = float(
            kind == "jepa_prior" or kind.startswith("jepa_visreg")
        )
        cfg.objective.vicreg = float(not kind.startswith("jepa_visreg"))
        cfg.objective.visreg = float(kind.startswith("jepa_visreg"))
        cfg.objective.gar = float(kind not in {
            "jepa_visreg_d2_nogar", "jepa_visreg_d1_nogar",
        })
        loss_fn = lambda module: compute_losses(
            jepa_forward(module, batch, device), cfg, module, batch,
        )[0]
    return model, batch, loss_fn


def main():
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    result = {
        "kind": args.kind, "microbatch": args.batch_size,
        "gradient_accumulation_steps": args.accumulation,
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
    }
    try:
        model, batch, loss_fn = build(args.kind, args.batch_size, device)
        optimizer = build_optimizer(model, lr=2e-3, weight_decay=0.05, betas=(0.9, 0.98))
        result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["optimizer_fused"] = bool(optimizer.defaults.get("fused", False))
        result["input_shapes"] = {
            key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)
        }
        model.train()
        durations = []
        torch.cuda.reset_peak_memory_stats()
        if args.accumulation < 1:
            raise ValueError("accumulation must be positive")
        for index in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.accumulation):
                with torch.autocast("cuda", torch.bfloat16):
                    loss = loss_fn(model)
                (loss / args.accumulation).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if hasattr(model, "update_teacher"):
                model.update_teacher(0.99)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                durations.append(elapsed)
        seconds = sum(durations)
        result.update({
            "status": "ok", "loss": float(loss.detach()),
            "mean_optimizer_step_seconds": seconds / len(durations),
            "mean_microbatch_seconds": seconds / (
                len(durations) * args.accumulation
            ),
            "mean_step_seconds": seconds / len(durations),
            "examples_per_second": (
                args.batch_size * args.accumulation * len(durations) / seconds
            ),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })
    except torch.cuda.OutOfMemoryError as error:
        result.update({"status": "oom", "error": str(error)})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
