"""Collect one compiled-domain admission gate into a single pass/fail record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _strict(metrics) -> float | None:
    """Strict success: solved with zero excess actions."""
    if not metrics:
        return None
    by_excess = metrics.get("metrics_by_excess_actions", {})
    entry = by_excess.get("0")
    return None if entry is None else float(entry["success"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir

    validation = _load(root / "validation.json")
    horizon = _load(root / "horizon_loss.json")
    invariants = _load(root / "checkpoint_invariants.json")
    references = {
        name: _load(root / f"{name}_metrics.json")
        for name in ("random", "oracle")
    }
    cells = {}
    for name in ("aligned", "shuffled"):
        for interface in ("feasible_menu", "full"):
            cells[f"{name}_train_{interface}"] = _load(
                root / f"{name}_train_{interface}.json"
            )
    for interface in ("feasible_menu", "full"):
        cells[f"aligned_val_{interface}"] = _load(
            root / f"aligned_val_{interface}.json"
        )

    aligned = _strict(cells.get("aligned_train_feasible_menu"))
    shuffled = _strict(cells.get("shuffled_train_feasible_menu"))
    random_bound = _strict(references["random"])
    oracle_bound = _strict(references["oracle"])
    checks = {
        "schema_and_replay": bool(validation)
        and validation.get("split_identity_disjoint") is True
        and all(
            value.get("expert_catalogue_recall") == 1.0
            and value.get("exact_replay_rate") == 1.0
            and value.get("goal_success_rate") == 1.0
            for value in validation.get("splits", {}).values()
        ),
        "horizon_energy_active": bool(horizon)
        and horizon.get("horizon_ranking_active") is True,
        "oracle_bound_perfect": oracle_bound == 1.0,
        "random_bound_below_oracle": (
            random_bound is not None and oracle_bound is not None
            and random_bound < oracle_bound
        ),
        "tiny_overfit_beats_random": (
            aligned is not None and random_bound is not None
            and aligned > random_bound
        ),
        "action_shuffle_hurts": (
            aligned is not None and shuffled is not None
            and shuffled < aligned
        ),
        "dropout_zero_and_ema_eval": bool(invariants)
        and invariants.get("dropout_zero", invariants.get("all_dropout_zero"))
        is True,
        "closed_loop_runs": all(
            cells.get(f"aligned_val_{interface}") is not None
            for interface in ("feasible_menu", "full")
        ),
    }
    payload = {
        "domain": args.domain,
        "run_dir": str(root),
        "checks": checks,
        "all_passed": all(checks.values()),
        "numbers": {
            "random_strict_success": random_bound,
            "oracle_strict_success": oracle_bound,
            "tiny_aligned_train_strict_success": aligned,
            "tiny_shuffled_train_strict_success": shuffled,
            "tiny_aligned_val_strict_success": _strict(
                cells.get("aligned_val_feasible_menu")
            ),
            "tiny_aligned_val_full_catalogue_strict_success": _strict(
                cells.get("aligned_val_full")
            ),
        },
        "validation": validation,
        "checkpoint_invariants": invariants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["checks"], indent=2, sort_keys=True))
    print(json.dumps(payload["numbers"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
