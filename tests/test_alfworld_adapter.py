import pytest

from textjepa.data.alfworld import (
    CATALOGUE_POLICY_VERSION,
    collect_alfworld_record,
    compile_alfworld_trace,
    observed_action_catalogue,
    observed_entities,
    split_task_observation,
)


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
    record = _trace()
    record["steps"][0]["admissible_commands"].append(
        "oracle command outside observed catalogue"
    )
    episode = compile_alfworld_trace(record, "train")
    assert "look" in episode.transitions[1].catalogue
    assert "look" not in episode.transitions[1].available
    assert "oracle command outside observed catalogue" not in (
        episode.transitions[0].available
    )
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


def test_observed_catalogue_uses_history_not_oracle_menu():
    initial = (
        "Looking quickly around you, you see a countertop 1, a fridge 1, "
        "a microwave 1, and a sinkbasin 1."
    )
    history = [initial, "On the countertop 1, you see an apple 1."]
    catalogue = observed_action_catalogue(initial, history)
    assert "take apple 1 from countertop 1" in catalogue
    assert "move apple 1 to fridge 1" in catalogue
    assert "heat apple 1 with microwave 1" in catalogue
    assert "clean apple 1 with sinkbasin 1" in catalogue
    # This builder receives no admissible-command argument.
    assert CATALOGUE_POLICY_VERSION == "observed-entities-v1"


def test_entity_and_task_parsing_are_deterministic():
    description = "You see an apple 1 and a mug 2. Your task is to: test."
    assert observed_entities(description) == ("apple 1", "mug 2")
    assert split_task_observation(
        "A room. Your task is to: put apple on table"
    ) == ("A room.", "put apple on table")


def test_zero_counterfactual_budget_never_executes_an_alternative(
    monkeypatch, tmp_path,
):
    class FakeSession:
        def __init__(self, *args, **kwargs):
            self.finished = False

        def reset(self):
            self.finished = False
            return (
                "You see a counter 1. Your task is to: inspect the counter",
                {
                    "won": False,
                    "admissible_commands": ["go to counter 1", "look"],
                    "extra.expert_plan": ["go to counter 1"],
                },
            )

        def step(self, action):
            assert action == "go to counter 1"
            self.finished = True
            return "You arrive at counter 1.", 1, True, {"won": True}

        def close(self):
            pass

    monkeypatch.setattr(
        "textjepa.data.alfworld.AlfworldTextSession", FakeSession
    )
    gamefile = tmp_path / "train" / "game.tw-pddl"
    gamefile.parent.mkdir()
    gamefile.write_text("{}")
    record = collect_alfworld_record(
        gamefile, tmp_path, "train", seed=1, counterfactual_k=0
    )
    assert record["steps"][0]["counterfactuals"] == []
