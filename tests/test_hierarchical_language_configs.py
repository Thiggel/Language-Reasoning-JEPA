from pathlib import Path

from hydra import compose, initialize_config_dir


def test_staged_hydra_configs_compose_at_global_package():
    with initialize_config_dir(
        config_dir=str((Path(__file__).parents[1] / "configs").resolve()),
        version_base=None,
    ):
        initial = compose(
            config_name="experiment/hierarchical_language_oracle_token"
        )
        nested = compose(
            config_name="experiment/hierarchical_language_nested_sentence"
        )
        waypoint = compose(
            config_name="experiment/hierarchical_language_oracle_waypoint"
        )
        full = compose(config_name="experiment/hierarchical_language_full")
        commutation = compose(
            config_name="experiment/hierarchical_language_dynamic_commutation"
        )
        macro = compose(
            config_name="experiment/hierarchical_language_macro_action"
        )
        value = compose(
            config_name="experiment/hierarchical_language_value_distillation"
        )
    assert initial.research.stage == "FLAT_ORACLE_TOKEN"
    assert not initial.model.config.enable_macro_actions
    assert not initial.model.config.enable_value
    assert nested.research.stage == "SENTENCE_JEPA"
    assert nested.research.architecture == (
        "nested_e0_e0_to_1_v2_window_exact"
    )
    assert nested.research.freeze_e0
    assert nested.research.freeze_p0
    assert nested.research.train_e0_to_1
    assert not nested.research.direct_sentence_teacher
    assert "alignment" not in nested.loss
    assert waypoint.research.stage == "ORACLE_WAYPOINT"
    assert waypoint.research.oracle_goal_kind == "next_sentence_boundary"
    assert full.research.stage == "FULL_HIERARCHY"
    assert full.model.config.enable_macro_actions
    assert full.model.config.enable_value
    assert commutation.research.stage == "DYNAMIC_COMMUTATION"
    assert commutation.loss.commutation > 0
    assert macro.research.stage == "MACRO_ACTION"
    assert macro.model.config.enable_macro_actions
    assert value.research.stage == "VALUE_DISTILLATION"
    assert value.model.config.enable_value
    assert "B" not in initial.planning
    assert full.planning.KV == [1, 2, 4, 8]
