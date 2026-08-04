import pytest

from scripts.build_flat_token_oracle_candidates import require_flat_oracle_stage
from scripts.run_hierarchical_oracle_evaluation_matrix import (
    SPLITS,
    TOKEN_HORIZONS,
    WORKER_HORIZONS,
)
from textjepa.training.hierarchical_language import ResearchStage


def test_flat_oracle_remains_available_for_nested_and_value_checkpoints():
    for stage in (
        ResearchStage.TOKEN_JEPA,
        ResearchStage.SENTENCE_JEPA,
        ResearchStage.DYNAMIC_COMMUTATION,
        ResearchStage.MACRO_ACTION,
        ResearchStage.VALUE_DISTILLATION,
    ):
        require_flat_oracle_stage(stage)


def test_flat_oracle_rejects_validation_only_stage():
    with pytest.raises(ValueError, match="at least a TOKEN_JEPA"):
        require_flat_oracle_stage(ResearchStage.DATA_VALIDATION)


def test_oracle_matrix_uses_the_normative_splits_and_depths():
    assert SPLITS == (
        "id_test", "near_length_ood", "far_length_ood",
        "structural_ood", "paraphrase_ood",
    )
    assert TOKEN_HORIZONS == (4, 8, 16)
    assert WORKER_HORIZONS == (8, 16, 32, 64)
