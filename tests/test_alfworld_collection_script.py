import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_alfworld_intent_data.py"
_SPEC = importlib.util.spec_from_file_location("collect_alfworld_intent_data", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _write_template(path, entries):
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


def test_template_games_preserve_exact_order_and_identity(tmp_path):
    root = tmp_path / "engine"
    relatives = ["json_2.1.1/train/b/game.tw-pddl", "json_2.1.1/train/a/game.tw-pddl"]
    for relative in relatives:
        game = root / relative
        game.parent.mkdir(parents=True, exist_ok=True)
        game.write_text("{}")
    template = tmp_path / "train.jsonl"
    _write_template(template, [
        {"split": "train", "metadata": {"gamefile_relative": relative}}
        for relative in relatives
    ])
    assert _MODULE._template_games(template, root, "train") == [
        root / relative for relative in relatives
    ]


def test_template_games_reject_duplicates_and_wrong_split(tmp_path):
    root = tmp_path / "engine"
    relative = "json_2.1.1/train/a/game.tw-pddl"
    game = root / relative
    game.parent.mkdir(parents=True)
    game.write_text("{}")
    template = tmp_path / "train.jsonl"
    _write_template(template, [
        {"split": "train", "metadata": {"gamefile_relative": relative}},
        {"split": "train", "metadata": {"gamefile_relative": relative}},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        _MODULE._template_games(template, root, "train")
    _write_template(template, [
        {"split": "val", "metadata": {"gamefile_relative": relative}},
    ])
    with pytest.raises(ValueError, match="non-train"):
        _MODULE._template_games(template, root, "train")
