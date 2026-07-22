"""Collect, replay, and compile the text-only ALFWorld paper dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import distribution, version
import json
import multiprocessing
from pathlib import Path
import random

from textjepa.data.alfworld import (
    ALFWORLD_SOURCE_REVISION,
    CATALOGUE_POLICY_VERSION,
    collect_alfworld_record,
    compile_alfworld_trace,
    replay_alfworld_record,
)


SPLITS = {
    "train": "train",
    "val": "valid_seen",
    "test": "valid_unseen",
}


def _solvable_games(root: Path) -> list[Path]:
    games = []
    for path in root.rglob("game.tw-pddl"):
        text = str(path)
        if "movable" in text or "Sliced" in text:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("solvable") is True:
            games.append(path)
    return sorted(games)


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _runtime_provenance() -> dict:
    direct_url = json.loads(
        distribution("alfworld").read_text("direct_url.json") or "{}"
    )
    installed_revision = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_revision != ALFWORLD_SOURCE_REVISION:
        raise RuntimeError(
            "ALFWorld collector requires pinned revision "
            f"{ALFWORLD_SOURCE_REVISION}, found {installed_revision!r}"
        )
    return {
        "alfworld_version": version("alfworld"),
        "alfworld_source_revision": installed_revision,
        "textworld_version": version("textworld"),
    }


def _collect_and_replay(payload: tuple) -> dict:
    """Collect one game in a disposable process.

    Fast Downward maps a private shared library for every TextWorld engine.
    Process isolation guarantees that mapping and its scratch file are
    reclaimed after every episode instead of accumulating over a dataset.
    """
    (
        gamefile, data_root, split, seed, counterfactual_k,
        teacher_horizon, max_steps, counterfactual_attempts,
    ) = payload
    record = collect_alfworld_record(
        gamefile, data_root, split, seed,
        counterfactual_k=counterfactual_k,
        teacher_horizon=teacher_horizon,
        max_steps=max_steps,
        counterfactual_attempts_per_step=counterfactual_attempts,
    )
    replay_alfworld_record(record, data_root)
    return record


def collect_split(args, split: str, limit: int) -> dict:
    source = args.data_root / "json_2.1.1" / SPLITS[split]
    games = _solvable_games(source)
    random.Random(f"{args.seed}:{split}").shuffle(games)
    records, failures = [], []
    context = multiprocessing.get_context("spawn")
    for gamefile in games:
        with context.Pool(processes=1, maxtasksperchild=1) as pool:
            payload = (
                gamefile, args.data_root, split, args.seed,
                args.counterfactual_k, args.teacher_horizon, args.max_steps,
                args.counterfactual_attempts,
            )
            try:
                pending = pool.apply_async(_collect_and_replay, (payload,))
                record = pending.get(timeout=args.episode_timeout_seconds)
                records.append(record)
            except multiprocessing.TimeoutError:
                failures.append({
                    "gamefile": str(gamefile),
                    "error": (
                        "episode timeout after "
                        f"{args.episode_timeout_seconds} seconds"
                    ),
                })
            except Exception as error:
                failure = {"gamefile": str(gamefile), "error": repr(error)}
                failures.append(failure)
                if "catalogue misses expert" in str(error):
                    _write_jsonl(
                        args.output / "failures" / f"{split}.jsonl", failures
                    )
                    raise RuntimeError(
                        "non-oracle catalogue recall is below 100%; dataset blocked"
                    ) from error
        if len(records) >= limit:
            break
    if len(records) < limit:
        _write_jsonl(args.output / "failures" / f"{split}.jsonl", failures)
        raise RuntimeError(
            f"ALFWorld {split}: requested {limit}, collected {len(records)} "
            f"from {len(games)} games ({len(failures)} failures)"
        )
    compiled = [asdict(compile_alfworld_trace(value, split)) for value in records]
    _write_jsonl(args.output / "raw" / f"{split}.jsonl", records)
    _write_jsonl(args.output / "compiled" / f"{split}.jsonl", compiled)
    _write_jsonl(args.output / "failures" / f"{split}.jsonl", failures)
    transitions = sum(len(value["steps"]) for value in records)
    counterfactuals = sum(
        len(step["counterfactuals"])
        for value in records for step in value["steps"]
    )
    return {
        "episodes": len(records), "transitions": transitions,
        "counterfactuals": counterfactuals, "failed_games": len(failures),
        "candidate_games": len(games),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--counterfactual-k", type=int, default=2)
    parser.add_argument("--teacher-horizon", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--counterfactual-attempts", type=int, default=4)
    parser.add_argument("--episode-timeout-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1741)
    parser.add_argument(
        "--split", choices=("all", "train", "val", "test"), default="all"
    )
    args = parser.parse_args()
    sizes = {
        "train": args.train_size, "val": args.val_size,
        "test": args.test_size,
    }
    summary = {
        "schema_version": 1,
        "catalogue_policy": CATALOGUE_POLICY_VERSION,
        "seed": args.seed,
        "counterfactual_k": args.counterfactual_k,
        "teacher_horizon": args.teacher_horizon,
        "counterfactual_attempts": args.counterfactual_attempts,
        "episode_timeout_seconds": args.episode_timeout_seconds,
        "runtime": _runtime_provenance(),
        "splits": {},
    }
    selected = sizes.items() if args.split == "all" else (
        (args.split, sizes[args.split]),
    )
    for split, size in selected:
        summary["splits"][split] = collect_split(args, split, size)
        print(split, summary["splits"][split], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
