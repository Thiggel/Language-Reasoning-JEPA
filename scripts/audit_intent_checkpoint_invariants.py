"""Assert the paper JEPA's dropout and EMA-target invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import nn

from textjepa.utils.checkpoint import load_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, _, cfg = load_run(args.checkpoint, args.device)
    model.train()
    dropout = [
        module.p for module in model.modules()
        if isinstance(module, nn.Dropout)
    ]
    teachers = [
        model.chunk_teacher.training,
        model.chunk_teacher.module.training,
        model.state_teacher.training,
        model.state_teacher.module.training,
    ]
    if any(value != 0.0 for value in dropout):
        raise RuntimeError(f"nonzero dropout probabilities: {dropout}")
    if any(teachers):
        raise RuntimeError(f"EMA teacher re-entered training mode: {teachers}")
    payload = {
        "checkpoint": args.checkpoint,
        "configured_dropout": float(cfg.model.dropout),
        "dropout_modules": len(dropout),
        "all_dropout_zero": True,
        "ema_wrapper_and_encoder_eval_after_parent_train": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
