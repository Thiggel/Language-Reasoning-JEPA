from pathlib import Path

from hydra import compose, initialize_config_dir

from scripts.run_hierarchical_value_continuation import CONFIGS


def _compose(name: str):
    root = Path("configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        return compose(config_name=f"experiment/{name.removesuffix('.yaml')}")


def test_every_continuation_variant_preserves_geometry_across_stages():
    expected = {
        "euclidean-vicreg": ("euclidean", "vicreg"),
        "mahalanobis-vicreg": ("mahalanobis", "vicreg"),
        "euclidean-sigreg": ("euclidean", "sigreg"),
    }
    for variant, configs in CONFIGS.items():
        geometry, regularizer = expected[variant]
        stages = ("DYNAMIC_COMMUTATION", "MACRO_ACTION", "VALUE_DISTILLATION")
        for filename, stage in zip(configs[:3], stages, strict=True):
            config = _compose(filename)
            assert config.research.stage == stage
            assert config.research.joint_token_sentence is False
            assert config.loss.dynamics_geometry == geometry
            assert config.loss.anti_collapse == regularizer


def test_value_continuations_enable_only_the_modules_needed_by_each_stage():
    for configs in CONFIGS.values():
        dynamic = _compose(configs[0])
        macro = _compose(configs[1])
        value = _compose(configs[2])
        assert not dynamic.model.config.get("enable_macro_actions", False)
        assert macro.model.config.enable_macro_actions is True
        assert not macro.model.config.get("enable_value", False)
        assert value.model.config.enable_macro_actions is True
        assert value.model.config.enable_value is True
