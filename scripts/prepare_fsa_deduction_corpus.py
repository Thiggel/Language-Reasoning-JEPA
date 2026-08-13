"""Build the length-controlled FSA logical-deduction corpus with OOD bands.

The band design follows the source project (``synthetic-RLVL``), where the
step-count bands are defined relative to the maximum training depth:
``band_train`` is ``step <= train_max``, ``band_ood`` is ``step > train_max``,
and ``band_hard_tail`` is the longest slice.  Here train/val are drawn from
15-26 deduction layers and the test ladder walks 15-26, 27-32, 33-40, 41-50.

Every split uses a disjoint sampling seed space, and identities are checked by
content hash so a resampled duplicate cannot cross splits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess

from textjepa.data.fsa_deduction import (
    canonical_identity,
    compile_fsa_episode,
    sample_fsa_problem,
)

BANDS: dict[str, tuple[int, int]] = {
    "train": (15, 26),
    "val": (15, 26),
    "test_id": (15, 26),
    "test_ood_27_32": (27, 32),
    "test_ood_33_40": (33, 40),
    "test_ood_41_50": (41, 50),
}
# Disjoint seed spaces per split of the deterministic sha256 index stream.
SEEDS: dict[str, int] = {
    "train": 100_000,
    "val": 200_000,
    "test_id": 300_000,
    "test_ood_27_32": 400_000,
    "test_ood_33_40": 500_000,
    "test_ood_41_50": 600_000,
}


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def build_split(
    name: str,
    count: int,
    *,
    split: str,
    branching_factor: int,
    teacher_horizon: int,
    counterfactual_k: int,
    catalogue_cap: int,
    side_facts_per_step: int,
    used: dict[str, str],
) -> tuple[list, Counter]:
    low, high = BANDS[name]
    rng = random.Random(SEEDS[name])
    episodes = []
    depths: Counter = Counter()
    index = 0
    while len(episodes) < count:
        depth = rng.randint(low, high)
        problem = sample_fsa_problem(
            seed=SEEDS[name], index=index, depth=depth,
            branching_factor=branching_factor,
            side_facts_per_step=side_facts_per_step,
        )
        index += 1
        identity = canonical_identity(problem)
        if identity in used:
            continue
        used[identity] = name
        episodes.append(compile_fsa_episode(
            problem, split,
            teacher_horizon=teacher_horizon,
            counterfactual_k=counterfactual_k,
            catalogue_cap=catalogue_cap,
        ))
        depths[depth] += 1
    return episodes, depths


def _write(path: Path, episodes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for episode in episodes:
            handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--val-episodes", type=int, default=300)
    parser.add_argument("--test-episodes", type=int, default=200)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--teacher-horizon", type=int, default=4)
    parser.add_argument("--counterfactual-k", type=int, default=3)
    parser.add_argument("--catalogue-cap", type=int, default=12)
    parser.add_argument(
        "--side-facts-per-step", type=int, default=0,
        help="0 reproduces the source recipe, where exactly one rule is "
             "applicable at every point; >0 widens the feasible menu.",
    )
    args = parser.parse_args()

    counts = {
        "train": args.train_episodes,
        "val": args.val_episodes,
        "test_id": args.test_episodes,
        "test_ood_27_32": args.test_episodes,
        "test_ood_33_40": args.test_episodes,
        "test_ood_41_50": args.test_episodes,
    }
    used: dict[str, str] = {}
    manifest_splits = {}
    for name, count in counts.items():
        split = "train" if name == "train" else "val" if name == "val" else "test"
        episodes, depths = build_split(
            name, count, split=split,
            branching_factor=args.branching_factor,
            teacher_horizon=args.teacher_horizon,
            counterfactual_k=args.counterfactual_k,
            catalogue_cap=args.catalogue_cap,
            side_facts_per_step=args.side_facts_per_step,
            used=used,
        )
        _write(args.out / f"{name}.jsonl", episodes)
        manifest_splits[name] = {
            "episodes": len(episodes),
            "schema_split": split,
            "depth_band": list(BANDS[name]),
            "sampling_seed": SEEDS[name],
            "transitions": sum(len(value.transitions) for value in episodes),
            "mean_transitions": round(
                sum(len(value.transitions) for value in episodes)
                / max(len(episodes), 1), 2,
            ),
            "mean_prompt_sentences": round(
                sum(len(value.prompt) for value in episodes)
                / max(len(episodes), 1), 2,
            ),
            "depth_histogram": dict(sorted(depths.items())),
        }
        print(f"{name}: {len(episodes)} episodes")

    root = Path(__file__).resolve().parents[1]
    # The default file names the training stack expects.
    (args.out / "test.jsonl").write_bytes(
        (args.out / "test_id.jsonl").read_bytes()
    )
    manifest = {
        "name": args.out.name,
        "domain": "fsa-deduction",
        "generator": "src/textjepa/data/fsa_deduction.py",
        "builder": "scripts/prepare_fsa_deduction_corpus.py",
        "git_sha": _git_sha(root),
        "provenance": {
            "source_repository": "alex:/home/hpc/c107fa/c107fa12/synthetic-RLVL",
            "source_file": "synthetic_dataset.py",
            "source_generator":
                "LogicDatasetGenerator._generate_hard_fsa_core (difficulty "
                "hard_fsa)",
            "source_docs": "docs/hfsa_depth_scaling_plan_2026-05-19.md",
            "licence":
                "owner's own unpublished research repository; no third-party "
                "licence applies. This is a re-implementation of the sampling "
                "recipe, not a code copy.",
            "band_convention":
                "synthetic-RLVL synthrlvl/eval_loop.py: band_train = step <= "
                "train_max, band_ood = step > train_max, band_hard_tail = "
                "longest slice.",
            "rendering":
                "natural-language rendering (their premises_nl/proof_nl "
                "form, selected in the source by task.template=natural); the "
                "source also emits a paired FOL rendering that is not used "
                "here.",
        },
        "design": {
            "length_knob": "depth = number of FSA layers; gold derivation has "
                           "2*depth - 1 inference steps",
            "branching_factor": args.branching_factor,
            "teacher_horizon": args.teacher_horizon,
            "counterfactual_k": args.counterfactual_k,
            "catalogue_cap": args.catalogue_cap,
            "side_facts_per_step": args.side_facts_per_step,
            "id_band": [15, 26],
            "ood_bands": [[27, 32], [33, 40], [41, 50]],
            "test_jsonl": "copy of test_id.jsonl so the standard "
                          "train/val/test triple is in-distribution; the OOD "
                          "bands are evaluated as separate test files.",
        },
        "splits": manifest_splits,
        "identity_check": "sha256 over initial facts, ground rules and target; "
                          "all splits disjoint by construction and verified",
    }
    (args.out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
