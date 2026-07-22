"""Audit compiled ALFWorld episodes and their non-oracle interactive replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textjepa.data.observed_action import load_observed_action_jsonl
from textjepa.planning.catalogue import environment_from_episode


def _parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be SPLIT=PATH")
    split, path = value.split("=", 1)
    if split not in {"train", "val", "test"}:
        raise argparse.ArgumentTypeError(f"unknown split: {split}")
    return split, Path(path)


def _require_counterfactual_coverage(
    episodes, admissible_k: int, invalid_k: int,
) -> None:
    for episode in episodes:
        for index, transition in enumerate(episode.transitions):
            admissible = sum(
                value.action in transition.available
                for value in transition.counterfactuals
            )
            invalid = sum(
                value.action not in transition.available
                for value in transition.counterfactuals
            )
            if admissible != admissible_k or invalid != invalid_k:
                raise RuntimeError(
                    f"{episode.episode_id} step {index}: counterfactual "
                    f"coverage admissible={admissible}/{admissible_k}, "
                    f"rejected={invalid}/{invalid_k}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=_parse_dataset,
                        required=True)
    parser.add_argument("--replay-limit", type=int, default=0,
                        help="zero replays every episode")
    parser.add_argument("--require-counterfactual-k", type=int, default=-1)
    parser.add_argument(
        "--require-invalid-counterfactual-k", type=int, default=-1
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    seen_ids: dict[str, str] = {}
    payload = {
        "schema_version": 1,
        "candidate_interface": "observed_entities_fixed_grammar",
        "oracle_menu_at_evaluation": False,
        "splits": {},
    }
    for split, path in args.dataset:
        episodes = load_observed_action_jsonl(
            path, expected_domain="alfworld-textworld"
        )
        if (
            args.require_counterfactual_k >= 0
            and args.require_invalid_counterfactual_k >= 0
        ):
            _require_counterfactual_coverage(
                episodes,
                args.require_counterfactual_k,
                args.require_invalid_counterfactual_k,
            )
        replayed = episodes[
            : args.replay_limit if args.replay_limit > 0 else len(episodes)
        ]
        transitions = sum(len(episode.transitions) for episode in episodes)
        counterfactuals = sum(
            len(transition.counterfactuals)
            for episode in episodes for transition in episode.transitions
        )
        invalid_counterfactuals = sum(
            value.action not in transition.available
            for episode in episodes for transition in episode.transitions
            for value in transition.counterfactuals
        )
        for episode in episodes:
            other = seen_ids.setdefault(episode.episode_id, split)
            if other != split:
                raise RuntimeError(
                    f"episode {episode.episode_id} overlaps {other} and {split}"
                )
        maximum_catalogue = 0
        for episode in replayed:
            environment = environment_from_episode(episode)
            try:
                environment.set_candidate_order_seed(
                    f"validation:{episode.episode_id}"
                )
                for index, transition in enumerate(episode.transitions):
                    catalogue = environment.catalogue
                    maximum_catalogue = max(maximum_catalogue, len(catalogue))
                    if transition.action not in catalogue:
                        raise RuntimeError(
                            f"{episode.episode_id} step {index}: dynamic "
                            "non-oracle catalogue misses the expert"
                        )
                    outcome = environment.step(transition.action)
                    if outcome != transition.outcome:
                        raise RuntimeError(
                            f"{episode.episode_id} step {index}: replay drift"
                        )
                if not environment.solved or environment.invalid_actions:
                    raise RuntimeError(
                        f"{episode.episode_id}: expert replay did not solve cleanly"
                    )
            finally:
                environment.close()
        payload["splits"][split] = {
            "episodes": len(episodes),
            "transitions": transitions,
            "counterfactuals": counterfactuals,
            "invalid_counterfactuals": invalid_counterfactuals,
            "full_counterfactual_coverage": (
                args.require_counterfactual_k >= 0
                and args.require_invalid_counterfactual_k >= 0
            ),
            "replayed_episodes": len(replayed),
            "expert_catalogue_recall": 1.0,
            "exact_replay_rate": 1.0,
            "goal_success_rate": 1.0,
            "maximum_dynamic_catalogue_size": maximum_catalogue,
        }
    payload["split_identity_disjoint"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
