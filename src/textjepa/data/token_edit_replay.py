"""Frozen-policy replay trajectories for faithful token-edit training.

Replay states and actions come from a target-free deployment policy.  The clean
solution is retained in a separately named privileged field solely to define
GAR teachers; it is never used to construct proposals or behavior actions.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from textjepa.data.faithful_token_edits import (
    OPS,
    _apply,
    _flat_length,
    _proposal_tokens,
    _render_action,
    propose_deployable_edits,
)
from textjepa.data.token_edit_distance import exact_one_step_advantages


REPLAY_FORMAT = "textjepa.faithful_token_edit_replay.v1"


def _copy_buffer(buffer: list[list[int]]) -> list[list[int]]:
    return [list(sentence) for sentence in buffer]


def _changed_sentence(
    before: list[list[int]], after: list[list[int]],
) -> list[int]:
    for prior, current in zip(before, after):
        if prior != current:
            return list(current)
    raise ValueError("replay action did not change its buffer")


class FrozenPolicyReplayDataset(Dataset):
    """Materialize exact transitions and fresh target-free proposal pools."""

    def __init__(
        self,
        path: str | Path,
        vocab,
        proposal_pool_k: int,
        proposal_token_pool: str = "prompt_plus_current",
        gar_teacher: str = "latent_distance",
        seed: int = 0,
    ):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != REPLAY_FORMAT:
            raise ValueError(f"unsupported replay format: {payload.get('format')}")
        self.path = str(Path(path).resolve())
        self.vocab = vocab
        self.records = payload.get("records", [])
        self.manifest = payload.get("manifest", {})
        self.proposal_pool_k = int(proposal_pool_k)
        self.proposal_token_pool = str(proposal_token_pool)
        self.gar_teacher = str(gar_teacher)
        self.seed = int(seed)
        if not self.records:
            raise ValueError("replay contains no valid trajectories")
        if self.proposal_pool_k < 1:
            raise ValueError("replay training requires proposal_pool_k > 0")
        if self.proposal_token_pool != "prompt_plus_current":
            raise ValueError(
                "first replay pilot requires deployment prompt_plus_current proposals"
            )
        if self.gar_teacher not in {"latent_distance", "token_edit_distance"}:
            raise ValueError(f"unknown gar_teacher: {self.gar_teacher}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index % len(self.records)]
        prompt = record["prompt"]
        buffers = [_copy_buffer(item) for item in record["buffer_snapshots"]]
        goal = _copy_buffer(record["terminal_privileged_goal_buffer"])
        raw_actions = [tuple(item) for item in record["behavior_actions"]]
        if len(buffers) != len(raw_actions) + 1 or not raw_actions:
            raise ValueError("replay trajectory must contain T actions and T+1 states")

        actions, operations, positions, content, changed = [], [], [], [], []
        proposal_actions, proposal_buffers, proposal_changed = [], [], []
        proposal_ops, proposal_positions, proposal_content = [], [], []
        gar_targets, gar_proposal_targets = [], []
        for step, action in enumerate(raw_actions):
            before, after = buffers[step], buffers[step + 1]
            verified = _copy_buffer(before)
            _apply(verified, action)
            if verified != after:
                raise ValueError("stored replay outcome is not the exact action result")
            kind, position, token = action
            actions.append(_render_action(self.vocab, action))
            operations.append(OPS[kind])
            positions.append(position)
            content.append(self.vocab.pad_id if token is None else int(token))
            changed.append(_changed_sentence(before, after))

            proposal_rng = random.Random(
                f"replay-proposals:{self.seed}:{record['snapshot_id']}:{step}"
            )
            tokens = _proposal_tokens(prompt, before, self.proposal_token_pool)
            candidates = propose_deployable_edits(
                before, tokens, self.proposal_pool_k, proposal_rng
            )
            step_actions, step_buffers, step_changed = [], [], []
            step_ops, step_positions, step_content = [], [], []
            for candidate in candidates:
                outcome = _copy_buffer(before)
                _apply(outcome, candidate)
                candidate_kind, candidate_position, candidate_token = candidate
                step_actions.append(_render_action(self.vocab, candidate))
                step_buffers.append(outcome)
                step_changed.append(_changed_sentence(before, outcome))
                step_ops.append(OPS[candidate_kind])
                step_positions.append(candidate_position)
                step_content.append(
                    self.vocab.pad_id
                    if candidate_token is None else int(candidate_token)
                )
            proposal_actions.append(step_actions)
            proposal_buffers.append(step_buffers)
            proposal_changed.append(step_changed)
            proposal_ops.append(step_ops)
            proposal_positions.append(step_positions)
            proposal_content.append(step_content)
            if self.gar_teacher == "token_edit_distance":
                advantages = exact_one_step_advantages(
                    before, [after, *step_buffers], goal, max_distance=None
                )
                gar_targets.append(advantages[0])
                gar_proposal_targets.append(advantages[1:])

        out = {
            "prompt": prompt,
            "buffers": buffers,
            "goal_buffer": goal,
            "actions": actions,
            "op": operations,
            "edit_position": positions,
            "edit_content_token": content,
            "value": [0] * len(raw_actions),
            "remaining": [0] * len(raw_actions),
            "resolved_n": [_flat_length(item) for item in buffers[1:]],
            "necessary": [1] * len(raw_actions),
            "answer": int(record["answer"]),
            "n_necessary": len(raw_actions),
            "n_vars": 0,
            "index": int(record["source_index"]),
            "edit_pos": [min(position, 15) for position in positions],
            "changed": changed,
            "defect_masks": [[] for _ in raw_actions],
            "proposal_actions": proposal_actions,
            "proposal_buffers": proposal_buffers,
            "proposal_changed": proposal_changed,
            "proposal_op": proposal_ops,
            "proposal_edit_position": proposal_positions,
            "proposal_edit_content_token": proposal_content,
            "replay_snapshot_id": record["snapshot_id"],
            "replay_problem_id": record["problem_id"],
            "information_regime": record["information_regime"],
        }
        if self.gar_teacher == "token_edit_distance":
            out["gar_token_edit_target"] = gar_targets
            out["gar_proposal_token_edit_target"] = gar_proposal_targets
        return out


class MixedReplayTokenEditDataset(Dataset):
    """Replace a fixed fraction of expert presentations with replay items."""

    def __init__(self, expert: Dataset, replay: FrozenPolicyReplayDataset,
                 fraction: float = 0.5):
        self.expert = expert
        self.replay = replay
        self.fraction = float(fraction)
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("replay_fraction must lie strictly between 0 and 1")
        self.replay_count = round(len(expert) * self.fraction)
        if self.replay_count < 1 or self.replay_count >= len(expert):
            raise ValueError("replay_fraction yields an empty expert or replay partition")
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.expert)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.expert, "set_epoch"):
            self.expert.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        position = index % len(self)
        if position < self.replay_count:
            replay_index = (self.epoch * self.replay_count + position) % len(
                self.replay
            )
            return self.replay[replay_index]
        item = self.expert[index]
        item["goal_buffer"] = _copy_buffer(item["buffers"][-1])
        return item


__all__ = [
    "REPLAY_FORMAT", "FrozenPolicyReplayDataset", "MixedReplayTokenEditDataset",
]
