"""Immutable provenance helpers for hierarchical-language artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update(digest: "hashlib._Hash", name: str, value: Any) -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        # NumPy has no native bfloat16 scalar type. Hash the exact underlying
        # storage bytes after recording dtype and shape above.
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    else:
        digest.update(json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8"))
    digest.update(b"\0")


def artifact_fingerprint(
    payload: dict[str, Any],
    fields: Iterable[str],
) -> str:
    """Hash named tensor/metadata fields in a stable, explicit order."""
    digest = hashlib.sha256()
    for name in sorted(fields):
        if name not in payload:
            raise ValueError(f"cannot fingerprint missing field: {name}")
        _update(digest, name, payload[name])
    return digest.hexdigest()
