"""Attach a separately trained state-value head to a compatible checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    value = torch.load(args.value, map_location="cpu", weights_only=False)
    prefix = "core.value_head."
    replacements = {
        name: tensor for name, tensor in value["model"].items()
        if name.startswith(prefix)
    }
    if not replacements:
        raise ValueError(f"{args.value} has no {prefix} tensors")
    for name, tensor in replacements.items():
        if name not in base["model"]:
            raise ValueError(f"base checkpoint lacks {name}")
        if base["model"][name].shape != tensor.shape:
            raise ValueError(
                f"shape mismatch for {name}: {base['model'][name].shape} "
                f"versus {tensor.shape}"
            )
        base["model"][name] = tensor
    base["component_sources"] = {
        **base.get("component_sources", {}),
        "core.value_head": str(Path(args.value)),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, destination)
    print(f"replaced {len(replacements)} tensors; saved {destination}")


if __name__ == "__main__":
    main()
