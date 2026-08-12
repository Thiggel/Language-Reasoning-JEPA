"""Build the paper PlanBench Blocksworld corpus with disjoint split identities.

The official PlanBench Blocksworld release contains only a few hundred
instances per set, and the earlier admission gate drew train, validation, and
test from a single directory.  This driver instead

1. pools every official instance from the requested instance sets,
2. checks identity **modulo block renaming** so a relabelled copy of a test
   problem cannot appear in training,
3. assigns official identities to validation and test first, keeping those
   splits purely official PlanBench problems,
4. tops the training split up with freshly sampled instances drawn from the
   same ``generated_basic`` recipe, rejecting any identity that collides with
   an already-used one, and
5. optionally compiles a longer-plan out-of-distribution test set.

Everything it needs to be reproduced is written to ``MANIFEST.json``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess

from textjepa.data.planbench import (
    action_catalogue,
    canonical_identity,
    compile_blocksworld_episode,
    load_blocksworld_pddl,
    random_blocksworld_problem,
    shortest_plan,
)


def _write(path: Path, episodes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _official_problems(
    source: Path, sets: list[str], blocks: set[int],
    dedupe: str = "canonical",
):
    """Load official instances, keyed by canonical identity.

    ``dedupe="canonical"`` keeps one instance per reasoning identity modulo
    block renaming.  ``dedupe="literal"`` keeps every distinct PDDL problem;
    the official ``generated`` plan-length set is built by relabelling one
    flat-to-tower task per block count, so it collapses to a single identity
    under canonical deduplication.
    """

    problems, provenance, rejected = {}, {}, Counter()
    for name in sets:
        directory = source / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(
            directory.glob("instance-*.pddl"),
            key=lambda value: int(value.stem.split("-")[1]),
        ):
            problem = load_blocksworld_pddl(path)
            if len(problem.objects) not in blocks:
                rejected["block_count"] += 1
                continue
            identity = canonical_identity(problem)
            if dedupe == "literal":
                identity = (
                    "literal", problem.objects, problem.initial, problem.goal
                )
            if identity in problems:
                rejected[
                    "literal_duplicate" if dedupe == "literal"
                    else "renaming_duplicate"
                ] += 1
                continue
            problems[identity] = problem
            provenance[identity] = f"{name}/{path.name}"
    return problems, provenance, dict(rejected)


def _plan_length(problem) -> int | None:
    plan = shortest_plan(
        problem.initial, problem.goal, action_catalogue(problem.objects)
    )
    return None if plan is None else len(plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planbench-root", type=Path, required=True,
                        help="directory holding the official instance sets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--instance-set", action="append", default=None,
        help="instance set directory name, repeatable",
    )
    parser.add_argument("--blocks", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--teacher-horizon", type=int, default=8)
    parser.add_argument("--counterfactual-k", type=int, default=-1)
    parser.add_argument("--invalid-counterfactual-k", type=int, default=2)
    parser.add_argument("--min-plan-length", type=int, default=2)
    parser.add_argument(
        "--ood-blocks", type=int, nargs="*", default=[6],
        help="block counts compiled into an extra longer-plan test file",
    )
    parser.add_argument("--ood-size", type=int, default=48)
    parser.add_argument("--ood-instance-set", default="generated")
    parser.add_argument("--seed", type=int, default=1741)
    args = parser.parse_args()

    sets = args.instance_set or [
        "generated_basic_3", "generated_basic", "generated"
    ]
    blocks = set(args.blocks)
    official, provenance, rejected = _official_problems(
        args.planbench_root, sets, blocks
    )
    order = sorted(official)
    rng = random.Random(args.seed)
    rng.shuffle(order)

    # Validation and test come first and are official-only, so every reported
    # PlanBench number is measured on official PlanBench instances.
    assignment: dict[str, list] = {"val": [], "test": [], "train": []}
    quotas = {
        "test": args.test_size, "val": args.val_size,
        "train": args.train_size,
    }
    used = set()
    plan_lengths: dict[str, Counter] = {
        split: Counter() for split in assignment
    }
    for split in ("test", "val", "train"):
        for identity in list(order):
            if len(assignment[split]) >= quotas[split]:
                break
            if identity in used:
                continue
            problem = official[identity]
            length = _plan_length(problem)
            if length is None or length < args.min_plan_length:
                used.add(identity)
                rejected["trivial_or_unsolvable"] = (
                    rejected.get("trivial_or_unsolvable", 0) + 1
                )
                continue
            used.add(identity)
            assignment[split].append((identity, problem, "official"))
            plan_lengths[split][length] += 1

    # Held-out reasoning identities.  A generated training instance is
    # rejected if it is isomorphic to any validation or test problem, which is
    # the leakage that matters; distinct block labellings of a *training*
    # identity are ordinary training data and are counted separately.
    held_out = {identity for split in ("val", "test")
                for identity, _, _ in assignment[split]}
    literal_train = {
        (problem.objects, problem.initial, problem.goal)
        for _, problem, _ in assignment["train"]
    }
    generated = 0
    generator = random.Random(f"{args.seed}:generated")
    block_choices = sorted(blocks)
    attempts = 0
    while len(assignment["train"]) < quotas["train"]:
        attempts += 1
        if attempts > 500 * quotas["train"]:
            raise RuntimeError("instance generator exhausted its attempt budget")
        n_blocks = block_choices[generator.randrange(len(block_choices))]
        problem = random_blocksworld_problem(
            generator, n_blocks, f"gen-{n_blocks}-{generated}"
        )
        literal = (problem.objects, problem.initial, problem.goal)
        if literal in literal_train:
            rejected["generated_literal_duplicate"] = (
                rejected.get("generated_literal_duplicate", 0) + 1
            )
            continue
        identity = canonical_identity(problem)
        if identity in held_out:
            rejected["generated_held_out_identity"] = (
                rejected.get("generated_held_out_identity", 0) + 1
            )
            continue
        length = _plan_length(problem)
        if length is None or length < args.min_plan_length:
            continue
        literal_train.add(literal)
        used.add(identity)
        generated += 1
        assignment["train"].append((identity, problem, "generated"))
        plan_lengths["train"][length] += 1

    classes = {
        split: {identity for identity, _, _ in items}
        for split, items in assignment.items()
    }
    for left in ("train", "val", "test"):
        for right in ("train", "val", "test"):
            if left < right and classes[left] & classes[right]:
                raise RuntimeError(
                    f"{left} and {right} share {len(classes[left] & classes[right])} "
                    "reasoning identities"
                )

    counts = {}
    for split, items in assignment.items():
        episodes = []
        for index, (identity, problem, origin) in enumerate(items):
            episode = compile_blocksworld_episode(
                problem, split, args.teacher_horizon,
                args.counterfactual_k, args.invalid_counterfactual_k,
            )
            source = provenance.get(identity, "generated")
            suffix = source.replace("/", "-").replace(".pddl", "")
            episode = type(episode)(
                episode_id=f"planbench-blocksworld-{split}-{index:05d}-{suffix}",
                domain=episode.domain,
                split=episode.split,
                prompt=episode.prompt,
                goal=episode.goal,
                transitions=episode.transitions,
                metadata={
                    **episode.metadata,
                    "instance_origin": origin,
                    "instance_source": source,
                    "blocks": len(problem.objects),
                },
            )
            episodes.append(episode)
        _write(args.output / f"{split}.jsonl", episodes)
        counts[split] = {
            "episodes": len(episodes),
            "official": sum(1 for item in items if item[2] == "official"),
            "generated": sum(1 for item in items if item[2] == "generated"),
            "transitions": sum(len(value.transitions) for value in episodes),
            "counterfactuals": sum(
                len(step.counterfactuals)
                for value in episodes for step in value.transitions
            ),
            "blocks": dict(sorted(Counter(
                value.metadata["blocks"] for value in episodes
            ).items())),
            "optimal_plan_length": dict(sorted(plan_lengths[split].items())),
            # Distinct reasoning identities modulo block renaming.  Reported
            # so nobody mistakes a relabelled training instance for a new
            # reasoning problem.
            "identity_classes": len({item[0] for item in items}),
        }
        print(split, counts[split], flush=True)

    if args.ood_blocks and args.ood_size:
        # Literal deduplication: see ``_official_problems``.  These problems
        # share one reasoning identity per block count by construction, and
        # that identity never occurs in the 3-5 block training corpus.
        ood, ood_provenance, _ = _official_problems(
            args.planbench_root, [args.ood_instance_set],
            set(args.ood_blocks), dedupe="literal",
        )
        ood_order = sorted(ood)
        random.Random(f"{args.seed}:ood").shuffle(ood_order)
        episodes, lengths = [], Counter()
        for identity in ood_order:
            if len(episodes) >= args.ood_size:
                break
            problem = ood[identity]
            if canonical_identity(problem) in classes["train"]:
                continue
            length = _plan_length(problem)
            if length is None or length < args.min_plan_length:
                continue
            used.add(identity)
            episode = compile_blocksworld_episode(
                problem, "test", args.teacher_horizon,
                args.counterfactual_k, args.invalid_counterfactual_k,
            )
            episodes.append(type(episode)(
                episode_id=(
                    "planbench-blocksworld-oodtest-"
                    f"{len(episodes):05d}-"
                    f"{ood_provenance[identity].replace('/', '-').replace('.pddl', '')}"
                ),
                domain=episode.domain, split="test",
                prompt=episode.prompt, goal=episode.goal,
                transitions=episode.transitions,
                metadata={
                    **episode.metadata, "instance_origin": "official",
                    "instance_source": ood_provenance[identity],
                    "blocks": len(problem.objects),
                    "length_ood": True,
                },
            ))
            lengths[length] += 1
        _write(args.output / "test_length_ood.jsonl", episodes)
        counts["test_length_ood"] = {
            "episodes": len(episodes),
            "official": len(episodes), "generated": 0,
            "transitions": sum(len(value.transitions) for value in episodes),
            "counterfactuals": sum(
                len(step.counterfactuals)
                for value in episodes for step in value.transitions
            ),
            "blocks": dict(sorted(Counter(
                value.metadata["blocks"] for value in episodes
            ).items())),
            "optimal_plan_length": dict(sorted(lengths.items())),
        }
        print("test_length_ood", counts["test_length_ood"], flush=True)

    manifest = {
        "dataset": "PlanBench Blocksworld observed-action episodes",
        "compiler": {
            "driver": "scripts/prepare_planbench_blocksworld_corpus.py",
            "module": "src/textjepa/data/planbench.py",
            "git_sha": _git_sha(Path(__file__).resolve().parent.parent),
            "python": ".venv/bin/python",
        },
        "options": {
            "instance_sets": sets,
            "blocks": sorted(blocks),
            "teacher_horizon": args.teacher_horizon,
            "counterfactual_k": args.counterfactual_k,
            "invalid_counterfactual_k": args.invalid_counterfactual_k,
            "min_plan_length": args.min_plan_length,
            "seed": args.seed,
            "ood_blocks": args.ood_blocks,
            "ood_instance_set": args.ood_instance_set,
        },
        "identity_policy": (
            "canonical identity is the (initial, goal) atom set minimised "
            "over all block renamings; train, validation, test, and the "
            "length-OOD test set are disjoint under that identity"
        ),
        "split_policy": (
            "validation and test contain official PlanBench instances only; "
            "training uses the remaining official instances plus freshly "
            "sampled generated_basic-recipe instances"
        ),
        "counts": counts,
        "rejected": rejected,
        "official_pool": len(official),
        "generated_train_instances": generated,
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
