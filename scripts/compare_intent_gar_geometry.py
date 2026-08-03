"""Compare matched GAR geometry audits and fit common frozen readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from textjepa.analysis.intent_decisions import (
    aggregate_rank_metrics,
    geometry_features,
    rank_metrics,
)
from textjepa.analysis.representations import effective_rank, linear_cka


def _parse(specification: str) -> tuple[str, Path, Path, Path]:
    try:
        label, train, test, audit = specification.split("=", 1)[0], *specification.split("=", 1)[1].split(",")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "model must be LABEL=TRAIN_NPZ,TEST_NPZ,AUDIT_JSON"
        ) from error
    return label, Path(train), Path(test), Path(audit)


def _unique_states(features) -> np.ndarray:
    _, indices = np.unique(features["group"], return_index=True)
    return features["current_online"][np.sort(indices)]


def _group_rank(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> dict:
    rows = []
    for group in np.unique(groups):
        selected = groups == group
        rows.append(rank_metrics(-probability[selected], y[selected]))
    return aggregate_rank_metrics(rows)


def _fit_probe(train, test, state_key: str, seed: int, forced: bool) -> dict:
    train_mask = train["forced"] == forced
    test_mask = test["forced"] == forced
    train_x = geometry_features(
        train[state_key][train_mask], train["goal_teacher"][train_mask]
    )
    test_x = geometry_features(
        test[state_key][test_mask], test["goal_teacher"][test_mask]
    )
    train_y = train["necessary"][train_mask].astype(bool)
    test_y = test["necessary"][test_mask].astype(bool)
    groups = train["group"][train_mask]
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    held_out = set(unique[:max(1, len(unique) // 5)].tolist())
    validation = np.asarray([group in held_out for group in groups])
    best = None
    for coefficient in (0.01, 0.1, 1.0, 10.0):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=coefficient, class_weight="balanced", max_iter=2000,
                random_state=seed,
            ),
        )
        estimator.fit(train_x[~validation], train_y[~validation])
        probability = estimator.predict_proba(train_x[validation])[:, 1]
        score = _group_rank(
            train_y[validation], probability, groups[validation]
        )["pairwise_accuracy"]
        candidate = (float(score or 0.0), -coefficient, estimator)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    coefficient = -best[1]
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=coefficient, class_weight="balanced", max_iter=2000,
            random_state=seed,
        ),
    )
    estimator.fit(train_x, train_y)
    probability = estimator.predict_proba(test_x)[:, 1]
    return {
        "selected_C": coefficient,
        "train_candidates": int(len(train_y)),
        "test_candidates": int(len(test_y)),
        "test": _group_rank(test_y, probability, test["group"][test_mask]),
    }


def _score_path(audit: dict, score: str, trace: str = "oracle") -> dict:
    return audit["summary"]["by_trace"][trace]["scores"][score]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", required=True,
        help="LABEL=TRAIN_NPZ,TEST_NPZ,AUDIT_JSON",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7321)
    args = parser.parse_args()
    loaded = {}
    result = {
        "interpretation": {
            "gar_teacher_geometry": "EMA true-next geometry; oracle transition",
            "online_oracle_geometry": "online true-next geometry; oracle transition",
            "predicted_geometry": "head-free latent imagination",
            "oracle_transition_value": "value readout with transition drift removed",
            "predicted_transition_value": "deployed JEPA one-step score",
            "probe": (
                "Same goal-relative logistic readout and C grid for every model; "
                "symbolic labels train the frozen diagnostic only."
            ),
        },
        "models": {},
        "linear_cka_current_online": {},
    }
    for specification in args.model:
        label, train_path, test_path, audit_path = _parse(specification)
        train = np.load(train_path)
        test = np.load(test_path)
        audit = json.loads(audit_path.read_text())
        loaded[label] = (train, test, audit)
        teacher = _score_path(audit, "gar_teacher_geometry")
        online = _score_path(audit, "online_oracle_geometry")
        predicted_geometry = _score_path(audit, "predicted_geometry")
        oracle_value = _score_path(audit, "oracle_transition_value")
        predicted_value = _score_path(audit, "predicted_transition_value")
        current = _unique_states(test)
        model_result = {
            "checkpoint": audit["checkpoint"],
            "native_path": {
                "gar_teacher_geometry": teacher,
                "online_oracle_geometry": online,
                "predicted_geometry": predicted_geometry,
                "oracle_transition_value": oracle_value,
                "predicted_transition_value": predicted_value,
            },
            "top1_localization_gaps": {
                "ema_to_online": (
                    online["competitive_top1"] - teacher["competitive_top1"]
                ),
                "online_true_to_predicted_geometry": (
                    predicted_geometry["competitive_top1"]
                    - online["competitive_top1"]
                ),
                "online_geometry_to_oracle_value": (
                    oracle_value["competitive_top1"]
                    - online["competitive_top1"]
                ),
                "oracle_value_to_deployed_value": (
                    predicted_value["competitive_top1"]
                    - oracle_value["competitive_top1"]
                ),
            },
            "representation": {
                "current_online_effective_rank": effective_rank(current),
            },
            "transition": audit["summary"]["transition"],
            "geometry": audit["summary"]["by_trace"]["oracle"]["geometry"],
            "common_frozen_readout": {
                trace: {
                    "true_next_teacher": _fit_probe(
                        train, test, "true_next_teacher", args.seed, forced
                    ),
                    "predicted_next": _fit_probe(
                        train, test, "predicted_next", args.seed, forced
                    ),
                }
                for trace, forced in (("oracle", False), ("forced_error", True))
            },
        }
        if "forced_error" in audit["summary"]["by_trace"]:
            model_result["forced_error"] = audit["summary"]["by_trace"][
                "forced_error"
            ]
        result["models"][label] = model_result
    for left, (_, left_test, _) in loaded.items():
        result["linear_cka_current_online"][left] = {}
        left_group = left_test["group"]
        for right, (_, right_test, _) in loaded.items():
            aligned = (
                np.array_equal(left_group, right_test["group"])
                and np.array_equal(left_test["action"], right_test["action"])
                and np.array_equal(left_test["forced"], right_test["forced"])
            )
            if not aligned:
                result["linear_cka_current_online"][left][right] = None
                continue
            result["linear_cka_current_online"][left][right] = linear_cka(
                _unique_states(left_test), _unique_states(right_test)
            )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
