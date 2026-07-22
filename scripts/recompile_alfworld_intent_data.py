"""Recompile preserved raw ALFWorld records after schema-only repairs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from textjepa.data.alfworld import compile_alfworld_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("raw ALFWorld recovery input is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in records:
            episode = compile_alfworld_trace(record, args.split)
            handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")
    print(json.dumps({"split": args.split, "episodes": len(records)}))


if __name__ == "__main__":
    main()
