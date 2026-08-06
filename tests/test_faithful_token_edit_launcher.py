from __future__ import annotations

import os
import subprocess
import importlib.util
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

from textjepa.objectives import GoalAdvantageDistill


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SPEC = importlib.util.spec_from_file_location("textjepa_train", ROOT / "scripts/train.py")
assert TRAIN_SPEC is not None and TRAIN_SPEC.loader is not None
TRAIN_MODULE = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(TRAIN_MODULE)


def test_structured_gar_config_builds_requested_objective() -> None:
    overrides = [
        "+experiment=edit_token_structured_ema_gar_h1",
        "objective.gar_action_value.weight=0.1",
        "objective.gar_action_value.regression_weight=0.25",
        "objective.gar_action_value.pairwise_weight=1",
        "train.lr=0.01",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    objective = TRAIN_MODULE.build_objective(cfg)
    gar = objective.objectives["gar_action_value"]
    assert isinstance(gar, GoalAdvantageDistill)
    assert objective.weights["gar_action_value"] == 0.1
    assert gar.regression_weight == 0.25
    assert gar.pairwise_weight == 1.0
    assert cfg.train.lr == 0.01


def test_launcher_forwards_training_overrides(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    call_log = tmp_path / "calls.log"
    env = os.environ | {
        "TEXTJEPA_ROOT": str(ROOT),
        "RUN_DIR": str(tmp_path / "run"),
        "RUN_ID": "launcher-test",
        "CALL_LOG": str(call_log),
    }
    (tmp_path / "run").mkdir()

    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_faithful_token_edit_cell.sh"),
            str(fake_python),
            "edit_token_structured_ema_gar_h1",
            "0",
            "data.counterfactual_k=8",
            "train.lr=0.01",
        ],
        env=env,
        check=True,
    )

    train_call = call_log.read_text(encoding="utf-8").splitlines()[0]
    assert "data.counterfactual_k=8" in train_call
    assert "train.lr=0.01" in train_call
