"""Validate compiled observed-action datasets without domain-specific leakage.

This gate checks the information boundary shared by ProofWriter and
PlanBench: split identities are disjoint, the public non-oracle catalogue
contains every expert action, recorded outcomes replay exactly, and the
executor reaches every recorded goal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textjepa.data.observed_action import load_observed_action_jsonl
from textjepa.planning.catalogue import environment_from_episode


def _dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be SPLIT=PATH")
    split, raw_path = value.split("=", 1)
    if split not in {"train", "val", "test"}:
        raise argparse.ArgumentTypeError(f"unknown split: {split}")
    return split, Path(raw_path)


def validate(
    datasets: list[tuple[str, Path]],
    domain: str,
    replay_limit: int = 0,
) -> dict:
    seen: dict[str, str] = {}
    payload = {
        "schema_version": 1,
        "domain": domain,
        "candidate_interface": "non_oracle_full_catalogue",
        "oracle_menu_at_evaluation": False,
        "splits": {},
    }
    for split, path in datasets:
        episodes = load_observed_action_jsonl(path, expected_domain=domain)
        selected = episodes[:replay_limit] if replay_limit > 0 else episodes
        for episode in episodes:
            previous = seen.setdefault(episode.episode_id, split)
            if previous != split:
                raise RuntimeError(
                    f"episode {episode.episode_id} overlaps {previous} and {split}"
                )

        maximum_catalogue = 0
        minimum_catalogue = None
        for episode in selected:
            environment = environment_from_episode(episode)
            try:
                set_order = getattr(
                    environment, "set_candidate_order_seed", None
                )
                if set_order is not None:
                    set_order(f"admission:{episode.episode_id}")
                for index, transition in enumerate(episode.transitions):
                    catalogue = environment.catalogue
                    maximum_catalogue = max(maximum_catalogue, len(catalogue))
                    minimum_catalogue = (
                        len(catalogue) if minimum_catalogue is None
                        else min(minimum_catalogue, len(catalogue))
                    )
                    if transition.action not in catalogue:
                        raise RuntimeError(
                            f"{episode.episode_id} step {index}: public "
                            "catalogue misses expert action"
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
                close = getattr(environment, "close", None)
                if close is not None:
                    close()

        payload["splits"][split] = {
            "episodes": len(episodes),
            "transitions": sum(len(value.transitions) for value in episodes),
            "counterfactuals": sum(
                len(step.counterfactuals)
                for value in episodes for step in value.transitions
            ),
            "replayed_episodes": len(selected),
            "expert_catalogue_recall": 1.0,
            "exact_replay_rate": 1.0,
            "goal_success_rate": 1.0,
            "minimum_catalogue_size": minimum_catalogue or 0,
            "maximum_catalogue_size": maximum_catalogue,
        }
    payload["split_identity_disjoint"] = True
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset", action="append", type=_dataset, required=True)
    parser.add_argument("--replay-limit", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = validate(args.dataset, args.domain, args.replay_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
