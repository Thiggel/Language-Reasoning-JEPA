import pytest

from textjepa.data.proofwriter import (
    compile_proofwriter_episode,
    parse_fact,
    parse_rule,
    rule_applications,
    shortest_derivation,
)


def _record():
    return {
        "id": "toy-D2",
        "triples": {
            "triple1": {"text": "Bob is blue.",
                        "representation": '("Bob" "is" "blue" "+")'},
            "triple2": {"text": "Alice sees Bob.",
                        "representation": '("Alice" "sees" "Bob" "+")'},
        },
        "rules": {
            "rule1": {
                "text": "Blue people are kind.",
                "representation": '((("someone" "is" "blue" "+")) -> '
                                  '(("someone" "is" "kind" "+")))',
            },
            "rule2": {
                "text": "Kind people are round.",
                "representation": '((("someone" "is" "kind" "+")) -> '
                                  '(("someone" "is" "round" "+")))',
            },
        },
        "questions": {
            "Q1": {"question": "Bob is round.", "answer": True,
                   "representation": '("Bob" "is" "round" "+")'},
        },
    }


def test_rule_engine_derives_a_two_step_query():
    record = _record()
    facts = frozenset(
        parse_fact(value["representation"])
        for value in record["triples"].values()
    )
    rules = tuple(
        parse_rule(key, value) for key, value in record["rules"].items()
    )
    first = rule_applications(facts, rules)
    assert [application.conclusion for application in first] == [
        ("Bob", "is", "kind", "+")
    ]
    plan = shortest_derivation(
        facts, rules, ("Bob", "is", "round", "+")
    )
    assert plan is not None and len(plan) == 2


def test_compiled_proof_episode_separates_catalogue_and_availability():
    episode = compile_proofwriter_episode(_record(), "Q1", "train")
    assert len(episode.transitions) == 2
    assert len(episode.transitions[0].catalogue) == 2
    assert len(episode.transitions[0].available) == 1
    assert episode.transitions[-1].outcome == "Bob is round."


def test_compiler_can_observe_every_invalid_catalogue_consequence():
    episode = compile_proofwriter_episode(
        _record(), "Q1", "train", counterfactual_k=None,
        invalid_counterfactual_k=-1,
    )
    for transition in episode.transitions:
        alternatives = {
            item.action: item for item in transition.counterfactuals
        }
        assert set(alternatives) == set(transition.catalogue) - {
            transition.action
        }
        for action in set(transition.catalogue) - set(transition.available):
            assert "no fact is added" in alternatives[action].outcome


def test_unknown_or_trivial_questions_are_not_trajectory_examples():
    record = _record()
    record["questions"]["Q1"]["answer"] = None
    with pytest.raises(ValueError, match="unknown"):
        compile_proofwriter_episode(record, "Q1", "train")


def test_static_catalogue_keeps_groundings_after_conclusion_is_known():
    facts = frozenset({
        ("lion", "eats", "lion", "+"),
        ("tiger", "eats", "lion", "+"),
        ("lion", "eats", "tiger", "+"),
    })
    rule = parse_rule("rule", {
        "text": "If something eats the lion then the lion eats the tiger.",
        "representation": (
            '("someone" "eats" "lion" "+") -> '
            '("lion" "eats" "tiger" "+")'
        ),
    })
    assert rule_applications(facts, (rule,)) == ()
    catalogue = rule_applications(
        facts, (rule,), include_known_conclusions=True
    )
    assert len(catalogue) == 2
