"""Write a runtime data config for a compiled observed-action gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = {
        "name": "observed_action",
        "domain": args.domain,
        "train_path": str((args.root / "train.jsonl").resolve()),
        "val_path": str((args.root / "val.jsonl").resolve()),
        "test_path": str((args.root / "test.jsonl").resolve()),
        "train_seed": 11,
        "val_seed": 12,
        "test_seed": 13,
        "geo_rank_k": 4,
        "geo_rank_horizon": 2,
        "fresh_per_epoch": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(cfg), args.out)


if __name__ == "__main__":
    main()
