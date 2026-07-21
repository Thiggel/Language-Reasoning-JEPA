"""Compile external reasoning domains into the observed-action JSONL schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import zipfile

from textjepa.data.alfworld import compile_alfworld_trace
from textjepa.data.planbench import (
    compile_blocksworld_episode,
    load_blocksworld_pddl,
)
from textjepa.data.proofwriter import compile_proofwriter_episode


def _write(path: Path, episodes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")


def prepare_proofwriter(args) -> None:
    members = {
        "train": f"proofwriter-dataset-V2020.12.3/{args.world}/"
                 f"{args.train_depth}/meta-train.jsonl",
        "val": f"proofwriter-dataset-V2020.12.3/{args.world}/"
               f"{args.eval_depth}/meta-dev.jsonl",
        "test": f"proofwriter-dataset-V2020.12.3/{args.world}/"
                f"{args.eval_depth}/meta-test.jsonl",
    }
    limits = {"train": args.train_size, "val": args.val_size,
              "test": args.test_size}
    with zipfile.ZipFile(args.archive) as archive:
        for split, member in members.items():
            episodes, skipped = [], 0
            with archive.open(member) as source:
                for raw in source:
                    record = json.loads(raw)
                    candidates = [
                        (key, value) for key, value in record["questions"].items()
                        if value.get("answer") in {True, False}
                        and int(value.get("QDep", 0)) >= args.min_depth
                    ]
                    if not candidates:
                        continue
                    # One query per theory avoids pretending correlated
                    # questions are independent examples. Prefer the deepest
                    # available query; break ties deterministically.
                    maximum = max(int(value.get("QDep", 0))
                                  for _, value in candidates)
                    candidates = [item for item in candidates
                                  if int(item[1].get("QDep", 0)) == maximum]
                    rng = random.Random(f"{args.seed}:{record['id']}:{split}")
                    question_id = candidates[rng.randrange(len(candidates))][0]
                    try:
                        episode = compile_proofwriter_episode(
                            record, question_id, split,
                            teacher_horizon=args.teacher_horizon,
                            counterfactual_k=args.counterfactual_k,
                        )
                    except (ValueError, RuntimeError):
                        skipped += 1
                        continue
                    episodes.append(episode)
                    if len(episodes) >= limits[split]:
                        break
            if len(episodes) < limits[split]:
                raise RuntimeError(
                    f"ProofWriter {split}: requested {limits[split]} but "
                    f"compiled {len(episodes)} ({skipped} skipped)"
                )
            _write(Path(args.output) / f"{split}.jsonl", episodes)
            print(f"proofwriter {split}: {len(episodes)} episodes; "
                  f"{skipped} skipped", flush=True)


def prepare_planbench(args) -> None:
    roots = {
        "train": Path(args.train_dir),
        "val": Path(args.val_dir),
        "test": Path(args.test_dir),
    }
    limits = {"train": args.train_size, "val": args.val_size,
              "test": args.test_size}
    used_ids = set()
    for split, root in roots.items():
        episodes, skipped = [], 0
        for path in sorted(root.glob("instance-*.pddl")):
            problem = load_blocksworld_pddl(path)
            identity = (len(problem.objects), problem.initial, problem.goal)
            if identity in used_ids:
                continue
            try:
                episode = compile_blocksworld_episode(
                    problem, split, args.teacher_horizon,
                    args.counterfactual_k,
                )
            except ValueError:
                skipped += 1
                continue
            used_ids.add(identity)
            episodes.append(episode)
            if len(episodes) >= limits[split]:
                break
        if len(episodes) < limits[split]:
            raise RuntimeError(
                f"PlanBench {split}: requested {limits[split]} but compiled "
                f"{len(episodes)} ({skipped} skipped)"
            )
        _write(Path(args.output) / f"{split}.jsonl", episodes)
        print(f"planbench {split}: {len(episodes)} episodes; "
              f"{skipped} skipped", flush=True)


def prepare_alfworld(args) -> None:
    for split in ("train", "val", "test"):
        source = Path(args.input) / f"{split}.jsonl"
        episodes = []
        with source.open() as handle:
            for line in handle:
                if line.strip():
                    episodes.append(compile_alfworld_trace(json.loads(line), split))
        _write(Path(args.output) / f"{split}.jsonl", episodes)
        print(f"alfworld {split}: {len(episodes)} episodes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="domain", required=True)

    proof = subparsers.add_parser("proofwriter")
    proof.add_argument("--archive", required=True)
    proof.add_argument("--output", required=True)
    proof.add_argument("--world", default="OWA", choices=("OWA", "CWA"))
    proof.add_argument("--train-depth", default="depth-3")
    proof.add_argument("--eval-depth", default="depth-5")
    proof.add_argument("--min-depth", type=int, default=1)
    proof.add_argument("--train-size", type=int, default=20_000)
    proof.add_argument("--val-size", type=int, default=2_000)
    proof.add_argument("--test-size", type=int, default=5_000)
    proof.add_argument("--teacher-horizon", type=int, default=8)
    proof.add_argument("--counterfactual-k", type=int, default=8)
    proof.add_argument("--seed", type=int, default=1741)
    proof.set_defaults(function=prepare_proofwriter)

    plan = subparsers.add_parser("planbench")
    plan.add_argument("--train-dir", required=True)
    plan.add_argument("--val-dir", required=True)
    plan.add_argument("--test-dir", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--train-size", type=int, default=500)
    plan.add_argument("--val-size", type=int, default=100)
    plan.add_argument("--test-size", type=int, default=300)
    plan.add_argument("--teacher-horizon", type=int, default=8)
    plan.add_argument("--counterfactual-k", type=int, default=8)
    plan.set_defaults(function=prepare_planbench)

    alf = subparsers.add_parser("alfworld")
    alf.add_argument("--input", required=True)
    alf.add_argument("--output", required=True)
    alf.set_defaults(function=prepare_alfworld)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
