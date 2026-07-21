"""Domain-neutral observed-action trajectory contract.

External reasoning environments are compiled once into this strict JSONL
schema.  Training code sees only the prompt, executed intent, observed
outcome, and a current-state action catalogue.  Privileged continuation
rollouts are isolated under ``teacher_rollouts`` and are used only to build
explicitly labelled geometric-teacher targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Iterable

from torch.utils.data import Dataset

from textjepa.data.vocab import Vocab


@dataclass(frozen=True)
class Counterfactual:
    action: str
    outcome: str
    teacher_rollouts: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def from_dict(cls, item: dict) -> "Counterfactual":
        return cls(
            action=str(item["action"]),
            outcome=str(item["outcome"]),
            teacher_rollouts=tuple(
                tuple(str(step) for step in rollout)
                for rollout in item.get("teacher_rollouts", [])
            ),
        )


@dataclass(frozen=True)
class ObservedTransition:
    action: str
    outcome: str
    catalogue: tuple[str, ...]
    available: tuple[str, ...]
    counterfactuals: tuple[Counterfactual, ...] = ()

    @classmethod
    def from_dict(cls, item: dict) -> "ObservedTransition":
        catalogue = tuple(str(action) for action in item["catalogue"])
        transition = cls(
            action=str(item["action"]),
            outcome=str(item["outcome"]),
            catalogue=catalogue,
            available=tuple(
                str(action)
                for action in item.get("available", catalogue)
            ),
            counterfactuals=tuple(
                Counterfactual.from_dict(value)
                for value in item.get("counterfactuals", [])
            ),
        )
        transition.validate()
        return transition

    def validate(self) -> None:
        if not self.action.strip() or not self.outcome.strip():
            raise ValueError("executed action and outcome must be non-empty")
        if len(set(self.catalogue)) != len(self.catalogue):
            raise ValueError("action catalogue contains duplicates")
        if self.action not in self.catalogue:
            raise ValueError("executed action is absent from its catalogue")
        if len(set(self.available)) != len(self.available):
            raise ValueError("available actions contain duplicates")
        if any(action not in self.catalogue for action in self.available):
            raise ValueError("available action is absent from catalogue")
        if self.action not in self.available:
            raise ValueError("executed action is not currently available")
        alternatives = [item.action for item in self.counterfactuals]
        if self.action in alternatives:
            raise ValueError("executed action duplicated as a counterfactual")
        if len(set(alternatives)) != len(alternatives):
            raise ValueError("counterfactual actions contain duplicates")
        if any(action not in self.available for action in alternatives):
            raise ValueError("counterfactual action is not currently available")


@dataclass(frozen=True)
class ObservedActionEpisode:
    episode_id: str
    domain: str
    split: str
    prompt: tuple[str, ...]
    goal: str
    transitions: tuple[ObservedTransition, ...]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict) -> "ObservedActionEpisode":
        episode = cls(
            episode_id=str(item["episode_id"]),
            domain=str(item["domain"]),
            split=str(item["split"]),
            prompt=tuple(str(value) for value in item["prompt"]),
            goal=str(item["goal"]),
            transitions=tuple(
                ObservedTransition.from_dict(value)
                for value in item["transitions"]
            ),
            metadata=dict(item.get("metadata", {})),
        )
        episode.validate()
        return episode

    def validate(self) -> None:
        if not self.episode_id or not self.domain or not self.split:
            raise ValueError("episode identity, domain, and split are required")
        if not self.prompt or not all(value.strip() for value in self.prompt):
            raise ValueError("prompt must contain non-empty chunks")
        if not self.goal.strip():
            raise ValueError("goal must be non-empty")
        if not self.transitions:
            raise ValueError("episode must contain at least one transition")


def load_observed_action_jsonl(
    path: str | Path, expected_domain: str | None = None,
) -> list[ObservedActionEpisode]:
    episodes = []
    seen = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = ObservedActionEpisode.from_dict(json.loads(line))
            except Exception as error:
                raise ValueError(f"invalid episode at line {line_number}: {error}") from error
            if expected_domain is not None and episode.domain != expected_domain:
                raise ValueError(
                    f"line {line_number} has domain {episode.domain!r}, "
                    f"expected {expected_domain!r}"
                )
            if episode.episode_id in seen:
                raise ValueError(f"duplicate episode_id: {episode.episode_id}")
            seen.add(episode.episode_id)
            episodes.append(episode)
    if not episodes:
        raise ValueError("observed-action JSONL is empty")
    return episodes


def build_observed_action_vocab(
    episodes: Iterable[ObservedActionEpisode],
) -> Vocab:
    tokens = []
    for episode in episodes:
        tokens.extend(" ".join((*episode.prompt, episode.goal)).split())
        for transition in episode.transitions:
            tokens.extend(transition.action.split())
            tokens.extend(transition.outcome.split())
            for action in transition.catalogue:
                tokens.extend(action.split())
            for alternative in transition.counterfactuals:
                tokens.extend(alternative.outcome.split())
                for rollout in alternative.teacher_rollouts:
                    for outcome in rollout:
                        tokens.extend(outcome.split())
    return Vocab(tokens)


class ObservedActionDataset(Dataset):
    """Expose compiled episodes through the existing discourse-JEPA tensors."""

    def __init__(
        self,
        episodes: list[ObservedActionEpisode],
        vocab: Vocab,
        geo_rank_k: int = 0,
        geo_rank_horizon: int = 1,
        seed: int = 0,
    ):
        if not episodes:
            raise ValueError("episodes must be non-empty")
        self.episodes = episodes
        self.vocab = vocab
        self.geo_rank_k = max(0, int(geo_rank_k))
        self.geo_rank_horizon = max(1, int(geo_rank_horizon))
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[index]
        transitions = episode.transitions
        length = len(transitions)
        catalogue = tuple(dict.fromkeys(
            action
            for transition in transitions
            for action in transition.catalogue
        ))
        item = {
            "prompt": [self.vocab.encode(value) for value in episode.prompt],
            "steps": [self.vocab.encode(value.outcome) for value in transitions],
            "actions": [self.vocab.encode(value.action) for value in transitions],
            # Domain-neutral placeholders used only by legacy diagnostics;
            # paper objectives for external domains must not enable symbolic
            # value, operation, or necessity supervision.
            "op": [0] * length,
            "value": [0] * length,
            "remaining": list(range(length - 1, -1, -1)),
            "resolved_n": list(range(1, length + 1)),
            "necessary": [1] * length,
            "answer": 0,
            "n_necessary": length,
            "n_vars": len(catalogue),
            "index": index,
            "var_idx": list(range(length)),
            "query_idx": length - 1,
            "ancestors": list(range(length)),
            "action_candidate_tokens": [
                self.vocab.encode(action) for action in catalogue
            ],
            "action_feasible": [
                [int(action in transition.available)
                 for action in catalogue]
                for transition in transitions
            ],
        }
        if self.geo_rank_k:
            eligible = [
                position for position, transition in enumerate(transitions)
                if transition.counterfactuals
            ]
            if eligible:
                rng = random.Random(f"{self.seed}:{episode.episode_id}:geo")
                anchor = eligible[rng.randrange(len(eligible))]
                transition = transitions[anchor]
                alternatives = list(transition.counterfactuals)
                rng.shuffle(alternatives)
                alternatives = alternatives[:self.geo_rank_k]
                candidates = [
                    Counterfactual(
                        transition.action,
                        transition.outcome,
                        (tuple(value.outcome for value in transitions[
                            anchor + 1:anchor + self.geo_rank_horizon
                        ]),),
                    ),
                    *alternatives,
                ]
                item.update(
                    ga_t=anchor,
                    ga_horizon=self.geo_rank_horizon,
                    ga_alt_actions=[
                        self.vocab.encode(value.action)
                        for value in alternatives
                    ],
                    ga_alt_steps=[
                        self.vocab.encode(value.outcome)
                        for value in alternatives
                    ],
                    ga_rollout_steps=[
                        [
                            [self.vocab.encode(value.outcome)] + [
                                self.vocab.encode(outcome)
                                for outcome in rollout[
                                    : self.geo_rank_horizon - 1
                                ]
                            ]
                            for rollout in (
                                value.teacher_rollouts
                                or ((),)
                            )
                        ]
                        for value in candidates
                    ],
                )
        return item
