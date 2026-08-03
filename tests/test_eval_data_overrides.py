from omegaconf import OmegaConf

from textjepa.utils.checkpoint import apply_eval_data_overrides


def test_eval_data_overrides_are_opt_in():
    run_cfg = OmegaConf.create({"data": {
        "steps_range": [3, 9],
        "n_vars_range": [6, 12],
        "leaf_prob": 0.35,
    }})
    apply_eval_data_overrides(run_cfg, OmegaConf.create({
        "eval_steps_range": None,
        "eval_n_vars_range": None,
        "eval_leaf_prob": None,
        "eval_sample_max_tries": None,
        "eval_strict_steps_range": None,
    }))
    assert list(run_cfg.data.steps_range) == [3, 9]
    assert "sample_max_tries" not in run_cfg.data


def test_eval_data_overrides_define_an_exact_length_cell():
    run_cfg = OmegaConf.create({"data": {
        "steps_range": [3, 9],
        "n_vars_range": [6, 12],
        "leaf_prob": 0.35,
    }})
    apply_eval_data_overrides(run_cfg, OmegaConf.create({
        "eval_steps_range": [11, 11],
        "eval_n_vars_range": [12, 12],
        "eval_leaf_prob": 0.35,
        "eval_sample_max_tries": 20000,
        "eval_strict_steps_range": True,
    }))
    assert list(run_cfg.data.steps_range) == [11, 11]
    assert list(run_cfg.data.n_vars_range) == [12, 12]
    assert run_cfg.data.sample_max_tries == 20000
    assert run_cfg.data.strict_steps_range is True
