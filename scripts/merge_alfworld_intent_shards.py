"""Merge sharded ALFWorld collection output into the paper dataset.

Each shard collected a deterministic stripe of the seeded game order.  This
driver concatenates the shards, re-checks the identity boundaries that the
admission gate depends on (unique episodes, unique games, splits disjoint by
game identity), and writes a MANIFEST recording provenance for every shard.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    values = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, action="append",
                        required=True, help="directory holding shard-* dirs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True,
                        choices=("train", "val", "test"))
    args = parser.parse_args()

    manifest = {
        "dataset": "text-only ALFWorld observed-action episodes",
        "collector": {
            "driver": "scripts/collect_alfworld_intent_data.py",
            "launcher": "scripts/collect_alfworld_intent_shards.sh",
            "merger": "scripts/merge_alfworld_intent_shards.py",
            "module": "src/textjepa/data/alfworld.py",
            "python": "/vol/home-vol2/ml/laitenbf/.venv-alfworld/bin/python",
        },
        "shards": {},
        "counts": {},
    }
    games_by_split: dict[str, set[str]] = {}
    for split in args.split:
        raw, compiled, shard_notes = [], [], []
        for shard_root in args.shard_root:
            for shard_dir in sorted(shard_root.glob("shard-*")):
                raw_path = shard_dir / "data" / "raw" / f"{split}.jsonl"
                compiled_path = shard_dir / "data" / "compiled" / f"{split}.jsonl"
                if not raw_path.is_file():
                    continue
                shard_raw = _read_jsonl(raw_path)
                shard_compiled = _read_jsonl(compiled_path)
                if len(shard_raw) != len(shard_compiled):
                    raise RuntimeError(f"{shard_dir}: raw/compiled length mismatch")
                raw.extend(shard_raw)
                compiled.extend(shard_compiled)
                shard_manifest = shard_dir / "data" / "manifest.json"
                failures = shard_dir / "data" / "failures" / f"{split}.jsonl"
                shard_notes.append({
                    "shard": str(shard_dir),
                    "episodes": len(shard_raw),
                    "failures": (
                        len(_read_jsonl(failures)) if failures.is_file() else 0
                    ),
                    "manifest": (
                        json.loads(shard_manifest.read_text())
                        if shard_manifest.is_file() else None
                    ),
                })
        if not raw:
            raise RuntimeError(f"no {split} shards found")
        identifiers = [value["episode_id"] for value in raw]
        if len(set(identifiers)) != len(identifiers):
            raise RuntimeError(f"{split}: duplicate episode identities")
        games = [value["gamefile_relative"] for value in raw]
        if len(set(games)) != len(games):
            raise RuntimeError(f"{split}: duplicate games")
        games_by_split[split] = set(games)
        _write_jsonl(args.output / "raw" / f"{split}.jsonl", raw)
        _write_jsonl(args.output / "compiled" / f"{split}.jsonl", compiled)
        manifest["shards"][split] = shard_notes
        manifest["counts"][split] = {
            "episodes": len(raw),
            "transitions": sum(len(value["steps"]) for value in raw),
            "counterfactuals": sum(
                len(step["counterfactuals"])
                for value in raw for step in value["steps"]
            ),
            "invalid_counterfactuals": sum(
                branch["action"] not in step["admissible_commands"]
                for value in raw for step in value["steps"]
                for branch in step["counterfactuals"]
            ),
            "rollout_action_annotated_branches": sum(
                bool(branch.get("teacher_rollout_actions"))
                for value in raw for step in value["steps"]
                for branch in step["counterfactuals"]
            ),
            "task_types": dict(sorted(Counter(
                value["task_type"] for value in raw
            ).items())),
            "raw_sha256": _sha256(args.output / "raw" / f"{split}.jsonl"),
            "compiled_sha256": _sha256(
                args.output / "compiled" / f"{split}.jsonl"
            ),
        }
        print(split, manifest["counts"][split]["episodes"], "episodes",
              flush=True)

    overlaps = {}
    for left in games_by_split:
        for right in games_by_split:
            if left < right:
                shared = games_by_split[left] & games_by_split[right]
                overlaps[f"{left}|{right}"] = len(shared)
                if shared:
                    raise RuntimeError(f"{left} and {right} share games")
    manifest["split_game_overlap"] = overlaps
    manifest["splits_disjoint"] = True
    manifest["split_sources"] = {
        "train": "json_2.1.1/train",
        "val": "json_2.1.1/valid_seen",
        "test": "json_2.1.1/valid_unseen",
    }
    manifest["information_boundary"] = (
        "admissible_commands and the hand-coded expert plan are privileged "
        "collection labels; deployment uses the observation-grounded "
        "catalogue (observed-entities-v1), and collection fails if that "
        "catalogue omits the expert action"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
