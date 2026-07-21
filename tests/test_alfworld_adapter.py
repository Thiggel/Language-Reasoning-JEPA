import pytest

from textjepa.data.alfworld import compile_alfworld_trace


def _trace():
    return {
        "episode_id": "pick-apple-1",
        "initial_observation": "You are in a kitchen.",
        "task": "put the apple on the table",
        "won": True,
        "steps": [{
            "catalogue": ["look", "go to counter 1"],
            "admissible_commands": ["look", "go to counter 1"],
            "expert_action": "go to counter 1",
            "next_observation": "You arrive at counter 1 and see an apple.",
        }, {
            "catalogue": ["look", "take apple 1 from counter 1"],
            "admissible_commands": ["take apple 1 from counter 1"],
            "expert_action": "take apple 1 from counter 1",
            "next_observation": "You take apple 1.",
        }],
    }


def test_alfworld_trace_keeps_nonoracle_catalogue_separate():
    episode = compile_alfworld_trace(_trace(), "train")
    assert "look" in episode.transitions[1].catalogue
    assert "look" not in episode.transitions[1].available
    assert episode.goal == "put the apple on the table"


def test_alfworld_trace_rejects_catalogue_recall_failure():
    record = _trace()
    record["steps"][0]["catalogue"] = ["look"]
    with pytest.raises(ValueError, match="misses expert"):
        compile_alfworld_trace(record, "train")


def test_alfworld_trace_rejects_failed_expert():
    record = _trace()
    record["won"] = False
    with pytest.raises(ValueError, match="does not satisfy"):
        compile_alfworld_trace(record, "train")
