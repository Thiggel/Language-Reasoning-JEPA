"""Assemble validated ALFWorld gate artifacts into a pilot-only dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from textjepa.data.observed_action import load_observed_action_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or parts[0] not in {"train", "val", "test"}:
        raise argparse.ArgumentTypeError("source must be SPLIT=COMPILED,RAW")
    paths = parts[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("source must include compiled and raw")
    return parts[0], Path(paths[0]), Path(paths[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=_source, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if {split for split, _, _ in args.source} != {"train", "val", "test"}:
        raise ValueError("exactly one train, val, and test source is required")

    seen: dict[str, str] = {}
    manifest = {
        "schema_version": 1,
        "purpose": "admission-pilot-only-not-headline-data",
        "splits_disjoint": True,
        "splits": {},
    }
    for split, compiled, raw in args.source:
        episodes = load_observed_action_jsonl(
            compiled, expected_domain="alfworld-textworld"
        )
        for episode in episodes:
            previous = seen.setdefault(episode.episode_id, split)
            if previous != split:
                raise ValueError(
                    f"episode {episode.episode_id} overlaps {previous} and {split}"
                )
            if episode.split != split:
                raise ValueError(
                    f"episode {episode.episode_id} declares split {episode.split}"
                )
        compiled_out = args.output / "compiled" / f"{split}.jsonl"
        raw_out = args.output / "raw" / f"{split}.jsonl"
        compiled_out.parent.mkdir(parents=True, exist_ok=True)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(compiled, compiled_out)
        shutil.copyfile(raw, raw_out)
        manifest["splits"][split] = {
            "episodes": len(episodes),
            "transitions": sum(len(ep.transitions) for ep in episodes),
            "counterfactuals": sum(
                len(transition.counterfactuals)
                for episode in episodes
                for transition in episode.transitions
            ),
            "invalid_counterfactuals": sum(
                alternative.action not in transition.available
                for episode in episodes
                for transition in episode.transitions
                for alternative in transition.counterfactuals
            ),
            "compiled_source": str(compiled.resolve()),
            "compiled_sha256": _sha256(compiled_out),
            "raw_source": str(raw.resolve()),
            "raw_sha256": _sha256(raw_out),
        }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
