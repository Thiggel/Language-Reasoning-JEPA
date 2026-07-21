"""Run preregistered held-out probes and geometry metrics on NPZ exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from textjepa.analysis.representations import (
    cluster_alignment,
    effective_rank,
    fit_categorical_probe,
    fit_numeric_probe,
    linear_cka,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--compare", help="aligned second test export for CKA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train, test = np.load(args.train), np.load(args.test)
    train_x, test_x = train["representations"], test["representations"]
    result = {
        "train_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "feature_dim": int(train_x.shape[1]),
        "effective_rank_train": effective_rank(train_x),
        "effective_rank_test": effective_rank(test_x),
        "numeric_probes": {}, "categorical_probes": {}, "clusters": {},
    }
    for key in sorted(name for name in train.files if name.startswith("numeric_")):
        if np.unique(train[key]).size < 2:
            continue
        probe = fit_numeric_probe(
            train_x, train[key], test_x, test[key], args.seed,
            train_groups=train["problem_id"],
        )
        result["numeric_probes"][key.removeprefix("numeric_")] = {
            "ridge_alpha": probe.hyperparameter,
            "validation_negative_mae": probe.validation_score,
            **probe.test_metrics,
        }
    for key in sorted(name for name in train.files if name.startswith("categorical_")):
        if np.unique(train[key]).size < 2:
            continue
        probe = fit_categorical_probe(
            train_x, train[key], test_x, test[key], args.seed,
            train_groups=train["problem_id"],
        )
        label = key.removeprefix("categorical_")
        result["categorical_probes"][label] = {
            "logistic_c": probe.hyperparameter,
            "validation_balanced_accuracy": probe.validation_score,
            **probe.test_metrics,
        }
        result["clusters"][label] = cluster_alignment(
            test_x, test[key], args.seed
        )
    if args.compare:
        other = np.load(args.compare)
        if not np.array_equal(test["problem_id"], other["problem_id"]):
            raise ValueError("CKA exports are not aligned by problem and step")
        result["linear_cka"] = linear_cka(test_x, other["representations"])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
