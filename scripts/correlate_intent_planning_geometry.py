"""Correlate local GAR geometry diagnostics with strict planning success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from textjepa.analysis.representations import effective_rank


def _parse(specification: str) -> tuple[str, str, list[Path | None]]:
    label, payload = specification.split("=", 1)
    fields = payload.split(",")
    if len(fields) < 3 or len(fields) > 5:
        raise argparse.ArgumentTypeError(
            "point must be LABEL=SCORE,AUDIT,METRICS[,REPRESENTATION,FEATURES]"
        )
    score = fields.pop(0)
    paths = [None if value == "-" else Path(value) for value in fields]
    paths.extend([None] * (4 - len(paths)))
    return label, score, paths


def _success(metrics: dict) -> float:
    if "metrics_by_slack" in metrics:
        return float(metrics["metrics_by_slack"]["0"]["success"])
    if "success" in metrics:
        return float(metrics["success"])
    candidates = [value for value in metrics.values() if isinstance(value, dict)]
    if len(candidates) == 1 and "success" in candidates[0]:
        return float(candidates[0]["success"])
    raise ValueError("could not locate strict success in metrics JSON")


def _optional(mapping: dict, *keys: str) -> float | None:
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value) if value is not None and np.isfinite(value) else None


def _point(specification: str) -> dict:
    label, score, (audit_path, metrics_path, rep_path, features_path) = _parse(
        specification
    )
    audit = json.loads(audit_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    ranking = audit["summary"]["by_trace"]["oracle"]["scores"][score]
    row = {
        "label": label,
        "score": score,
        "strict_success": _success(metrics),
        "pairwise_ranking": ranking.get("exact_ordering_accuracy", ranking.get("pairwise_accuracy")),
        "top1_optimal_recall": ranking.get("optimal_top1", ranking.get("competitive_top1")),
        "mean_action_regret": ranking.get("mean_regret"),
        "spearman_action_advantage": ranking.get("spearman"),
        "action_margin": ranking.get("optimal_margin_mean", ranking.get("margin_mean")),
        "one_step_latent_error": _optional(
            audit, "summary", "transition", "oracle", "layernorm_l1_mean"
        ),
        "within_state_retrieval": _optional(
            audit, "summary", "transition", "oracle", "within_state_retrieval"
        ),
    }
    if rep_path is not None:
        representation = json.loads(rep_path.read_text())
        row.update({
            "effective_rank": _optional(representation, "effective_rank_test"),
            "necessary_probe": _optional(
                representation, "categorical_probes", "necessary", "balanced_accuracy"
            ),
            "operation_probe": _optional(
                representation, "categorical_probes", "operation", "balanced_accuracy"
            ),
            "remaining_steps_probe": _optional(
                representation, "numeric_probes", "remaining_steps", "r2"
            ),
        })
    elif features_path is not None:
        features = np.load(features_path)
        _, indices = np.unique(features["group"], return_index=True)
        row["effective_rank"] = effective_rank(
            features["current_online"][np.sort(indices)]
        )
    return row


def _correlations(rows: list[dict]) -> dict:
    predictors = sorted(set().union(*(row.keys() for row in rows)) - {
        "label", "score", "strict_success"
    })
    result = {}
    for predictor in predictors:
        selected = [
            row for row in rows
            if row.get(predictor) is not None
            and np.isfinite(row[predictor])
            and np.isfinite(row["strict_success"])
        ]
        if len(selected) < 3:
            continue
        x = np.asarray([row[predictor] for row in selected], dtype=float)
        y = np.asarray([row["strict_success"] for row in selected], dtype=float)
        if x.std() == 0 or y.std() == 0:
            continue
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        result[predictor] = {
            "n": len(selected),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [_point(specification) for specification in args.point]
    result = {
        "claim": (
            "Local action ranking and margin should track strict planning "
            "better than global effective rank, probes, or one-step error."
        ),
        "rows": rows,
        "correlations_with_strict_success": _correlations(rows),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
