from pathlib import Path


def test_primary_oracle_cem_planner_has_no_auxiliary_language_model_dependency():
    source = Path("scripts/plan_token_hierarchy_oracle_cem.py").read_text()
    forbidden_dependencies = (
        "DecoderLM", "beam_spans", "proposal_lm", "plan_lm",
        "lm_baseline", "sentence_lm",
    )
    for dependency in forbidden_dependencies:
        assert dependency not in source
    assert "categorical_cem" in source
    assert "oracle_goal = model.teacher(full)[:, -1]" in source


def test_fast_matrix_crosses_every_support_mode_with_reachability():
    source = Path("scripts/run_hard_oracle_cem_fast_matrix.sh").read_text()
    for mode in (
        "unconstrained", "support_head", "global_bank", "conditional_bank",
        "gmm", "conditional_prior",
    ):
        assert mode in source
    assert "for reach in 0 1" in source
    assert "--macro-candidates 256 --macro-iterations 5" in source
