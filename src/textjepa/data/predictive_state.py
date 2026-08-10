"""Data contracts for action-conditioned predictive-state experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset


TOKEN_BLOCK_SCHEMA = "predictive_state_token_blocks_v1"
REASONING_STATE_SCHEMA = "predictive_state_reasoning_states_v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_fingerprint(texts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def pack_documents(
    token_documents: Sequence[Sequence[int]],
    *,
    context_length: int,
    eos_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pack documents with EOS boundaries and a boundary-safe target mask.

    The first token of a new document is never used as the target of the last
    token of the preceding document. Predicting EOS from document content
    remains valid. Incomplete trailing blocks are dropped so all cells see the
    same dense shapes.
    """
    if context_length < 2:
        raise ValueError("context_length must be at least two")
    stream: list[int] = []
    target_valid: list[bool] = []
    for document in token_documents:
        tokens = list(map(int, document))
        if not tokens:
            continue
        if stream:
            # The next appended token starts a document, so the transition
            # into it must not contribute to NTP or latent objectives.
            target_valid[-1] = False
        stream.extend(tokens)
        target_valid.extend([True] * len(tokens))
        if tokens[-1] != eos_token_id:
            stream.append(int(eos_token_id))
            target_valid.append(True)
    blocks = len(stream) // context_length
    if blocks < 1:
        raise ValueError("corpus is shorter than one context block")
    used = blocks * context_length
    input_ids = torch.tensor(stream[:used], dtype=torch.long).reshape(
        blocks, context_length
    )
    valid = torch.tensor(target_valid[:used], dtype=torch.bool).reshape(
        blocks, context_length
    )
    # target_mask[:, t] controls the x_t -> x_(t+1) transition.
    target_mask = valid[:, :-1].clone()
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "target_mask": target_mask,
    }


class TokenBlockDataset(Dataset):
    """Validated fixed-width token blocks saved by the preparation CLI."""

    def __init__(self, payload: dict[str, Any], split: str):
        if payload.get("schema_version") != TOKEN_BLOCK_SCHEMA:
            raise ValueError("unsupported predictive-state token schema")
        if split not in payload["splits"]:
            raise KeyError(f"missing split: {split}")
        data = payload["splits"][split]
        required = {"input_ids", "attention_mask", "target_mask"}
        if not required <= data.keys():
            raise ValueError("token block split is incomplete")
        inputs = data["input_ids"]
        attention = data["attention_mask"]
        target = data["target_mask"]
        if inputs.ndim != 2 or attention.shape != inputs.shape:
            raise ValueError("input_ids and attention_mask must be B x T")
        if target.shape != (inputs.shape[0], inputs.shape[1] - 1):
            raise ValueError("target_mask must align adjacent token pairs")
        self.data = data

    def __len__(self) -> int:
        return int(self.data["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.data.items()}


def load_token_blocks(path: Path, split: str) -> tuple[TokenBlockDataset, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return TokenBlockDataset(payload, split), payload["metadata"]


def save_token_blocks(
    path: Path,
    *,
    train: dict[str, torch.Tensor],
    validation: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": TOKEN_BLOCK_SCHEMA,
        "metadata": metadata,
        "splits": {"train": train, "validation": validation},
    }, path)


@dataclass(frozen=True)
class ReasoningStateRecord:
    problem_id: str
    prompt_state: torch.Tensor
    states: torch.Tensor
    valid: torch.Tensor
    remaining_chunks: torch.Tensor
    correct: bool
    terminal_index: int
    trajectory_id: str


def validate_reasoning_bundle(payload: dict[str, Any]) -> None:
    """Validate Stage 3 tensors and mandatory oracle/provenance labels."""
    if payload.get("schema_version") != REASONING_STATE_SCHEMA:
        raise ValueError("unsupported reasoning-state schema")
    metadata = payload.get("metadata", {})
    required_metadata = {
        "model_id", "model_revision", "checkpoint_fingerprint",
        "oracle_terminal_states", "candidate_privileged_outcomes",
        "cross_project_information",
    }
    if not required_metadata <= metadata.keys():
        raise ValueError("reasoning bundle lacks information-scope labels")
    for record in payload.get("records", []):
        required = {
            "problem_id", "trajectory_id", "prompt_state", "states",
            "valid", "remaining_chunks", "correct", "terminal_index",
        }
        if not required <= record.keys():
            raise ValueError("reasoning state record is incomplete")
        states = record["states"]
        if states.ndim != 2 or record["valid"].shape != states.shape[:1]:
            raise ValueError("reasoning states must be chunks x width")
        if record["remaining_chunks"].shape != states.shape[:1]:
            raise ValueError("remaining-chunk targets do not align")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
