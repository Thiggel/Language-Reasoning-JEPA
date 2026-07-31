#!/usr/bin/env python3
"""Bind measured validity gates to an exact checkpoint and dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
from pathlib import Path


OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--admitted-stage", required=True)
    parser.add_argument("--dataset-fingerprint")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    measured_payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    measured = measured_payload.get("metrics", measured_payload)
    checkpoint_digest = sha256(args.checkpoint)
    if measured_payload.get("checkpoint_sha256") not in {
        None, checkpoint_digest
    }:
        raise ValueError("metrics were produced by another checkpoint")
    measured_dataset = measured_payload.get("dataset_fingerprint")
    if args.dataset_fingerprint and measured_dataset not in {
        None, args.dataset_fingerprint
    }:
        raise ValueError("metrics were produced on another dataset")
    dataset_fingerprint = args.dataset_fingerprint or measured_dataset
    if not dataset_fingerprint:
        raise ValueError("admission metrics must identify the dataset")
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if not isinstance(measured, dict) or not isinstance(gate, dict):
        raise ValueError("metrics and gate must be JSON objects")
    outcomes = {}
    for name, rule in gate.items():
        if name not in measured or rule.get("op") not in OPERATORS:
            raise ValueError(f"invalid or missing gate metric: {name}")
        value = float(measured[name])
        threshold = float(rule["value"])
        outcomes[name] = bool(
            OPERATORS[rule["op"]](value, threshold)
        )
    record = {
        "passed": all(outcomes.values()),
        "admitted_stage": args.admitted_stage,
        "metrics": {name: float(value) for name, value in measured.items()},
        "gate": gate,
        "gate_outcomes": outcomes,
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_dataset_fingerprint": dataset_fingerprint,
        "admitted_checkpoint_sha256": checkpoint_digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not record["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
