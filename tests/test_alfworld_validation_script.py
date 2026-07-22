import importlib.util
from pathlib import Path

import pytest

from textjepa.data.observed_action import (
    Counterfactual,
    ObservedActionEpisode,
    ObservedTransition,
)


_SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_alfworld_intent_data.py"
_SPEC = importlib.util.spec_from_file_location("validate_alfworld_intent_data", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _episode(counterfactuals):
    return ObservedActionEpisode(
        episode_id="episode",
        domain="alfworld-textworld",
        split="train",
        prompt=("room", "goal"),
        goal="goal",
        transitions=(ObservedTransition(
            action="expert",
            outcome="next",
            catalogue=("expert", "admissible", "rejected"),
            available=("expert", "admissible"),
            counterfactuals=tuple(counterfactuals),
        ),),
    )


def test_counterfactual_coverage_accepts_one_of_each_kind():
    episode = _episode([
        Counterfactual("admissible", "changed"),
        Counterfactual("rejected", "unchanged"),
    ])
    _MODULE._require_counterfactual_coverage([episode], 1, 1)


def test_counterfactual_coverage_rejects_missing_rejected_action():
    episode = _episode([Counterfactual("admissible", "changed")])
    with pytest.raises(RuntimeError, match="rejected=0/1"):
        _MODULE._require_counterfactual_coverage([episode], 1, 1)
