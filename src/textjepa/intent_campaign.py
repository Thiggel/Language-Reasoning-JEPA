"""Frozen design constants and selection rules for the intent paper campaign."""

from __future__ import annotations

from dataclasses import dataclass
import math


LEARNING_RATES = (
    5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3,
)
SELECTION_SEEDS = (0, 1, 2)
CONFIRMATION_SEEDS = (3, 4)
PAPER_SEEDS = SELECTION_SEEDS + CONFIRMATION_SEEDS
WIDTHS = (128, 256, 512)
DATASETS = ("igsm", "proofwriter", "planbench_blocksworld", "alfworld")
LEARNED_MODELS = (
    "token_lm",
    "sentence_lm",
    "sentence_latent_lm",
    "looped_token_lm",
    "looped_sentence_lm",
    "looped_sentence_latent_lm",
    "geometry_jepa",
)


@dataclass(frozen=True)
class LearningRateResult:
    learning_rate: float
    seed: int
    validation_success: float


def select_learning_rate(results: list[LearningRateResult]) -> float:
    """Select on validation mean only, requiring the complete frozen grid."""
    cells = {(value.learning_rate, value.seed): value for value in results}
    expected = {
        (learning_rate, seed)
        for learning_rate in LEARNING_RATES
        for seed in SELECTION_SEEDS
    }
    missing = expected - set(cells)
    extra = set(cells) - expected
    if missing or extra:
        raise ValueError(
            f"incomplete learning-rate grid: {len(missing)} missing, "
            f"{len(extra)} unexpected cells"
        )
    if any(
        not math.isfinite(value.validation_success)
        for value in cells.values()
    ):
        raise ValueError("learning-rate grid contains non-finite validation metrics")
    means = {
        learning_rate: sum(
            cells[learning_rate, seed].validation_success
            for seed in SELECTION_SEEDS
        ) / len(SELECTION_SEEDS)
        for learning_rate in LEARNING_RATES
    }
    # Frozen tie break: lower learning rate. This avoids selecting an unstable
    # high-rate endpoint based on floating-point noise.
    return max(LEARNING_RATES, key=lambda rate: (means[rate], -rate))


def campaign_counts() -> dict[str, int]:
    learned_cells = len(DATASETS) * len(LEARNED_MODELS) * len(WIDTHS)
    return {
        "learned_model_dataset_width_cells": learned_cells,
        "learning_rate_selection_trainings": (
            learned_cells * len(LEARNING_RATES) * len(SELECTION_SEEDS)
        ),
        "post_selection_confirmation_trainings": (
            learned_cells * len(CONFIRMATION_SEEDS)
        ),
        "five_seed_main_training_members": learned_cells * len(PAPER_SEEDS),
    }
