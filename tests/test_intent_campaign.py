import pytest

from textjepa.intent_campaign import (
    CONFIRMATION_SEEDS,
    DATASETS,
    LEARNED_MODELS,
    LEARNING_RATES,
    PAPER_SEEDS,
    SELECTION_SEEDS,
    WIDTHS,
    LearningRateResult,
    campaign_counts,
    select_learning_rate,
)


def _grid(best=7e-4):
    return [
        LearningRateResult(rate, seed, 0.8 if rate == best else 0.5)
        for rate in LEARNING_RATES for seed in SELECTION_SEEDS
    ]


def test_frozen_campaign_matches_human_requested_grid():
    assert LEARNING_RATES == (
        5e-5, 7e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3,
    )
    assert SELECTION_SEEDS == (0, 1, 2)
    assert CONFIRMATION_SEEDS == (3, 4)
    assert PAPER_SEEDS == (0, 1, 2, 3, 4)
    assert WIDTHS == (128, 256, 512)
    assert len(DATASETS) == 4 and len(LEARNED_MODELS) == 7
    assert campaign_counts()["learning_rate_selection_trainings"] == 2268


def test_lr_selection_uses_three_seed_mean_and_fixed_tie_break():
    assert select_learning_rate(_grid()) == 7e-4
    tied = [LearningRateResult(rate, seed, 0.5)
            for rate in LEARNING_RATES for seed in SELECTION_SEEDS]
    assert select_learning_rate(tied) == min(LEARNING_RATES)


def test_lr_selection_refuses_partial_or_nonfinite_sweeps():
    with pytest.raises(ValueError, match="incomplete"):
        select_learning_rate(_grid()[:-1])
    values = _grid()
    values[0] = LearningRateResult(
        values[0].learning_rate, values[0].seed, float("nan")
    )
    with pytest.raises(ValueError, match="non-finite"):
        select_learning_rate(values)
