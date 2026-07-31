from textjepa.data.igsm_step_verifier import (
    achieved_next_state_id,
    final_answer_matches,
    operation_matches_expected,
    parse_rendered_operation,
)


def test_parses_both_canonical_and_paraphrase_renderers():
    canonical = parse_rendered_operation(
        "so the number of purple bells is 7 plus 4 = 11 ."
    )
    paraphrase = parse_rendered_operation(
        "Computing purple bells: 7 plus 4 gives 11."
    )
    assert canonical == paraphrase


def test_verifier_accepts_commuted_addition_but_not_subtraction():
    assert operation_matches_expected(
        "purple bells is 4 plus 7 = 11",
        "so the number of purple bells is 7 plus 4 = 11 .",
    )
    assert not operation_matches_expected(
        "green pens is 4 minus 12 = 15",
        "so the number of green pens is 12 minus 4 = 8 .",
    )


def test_verifier_checks_modular_arithmetic_and_target_variable():
    expected = "so the number of red shells is 11 times 7 = 8 ."
    assert operation_matches_expected(
        "Computing red shells: 11 times 7 gives 8.", expected
    )
    assert not operation_matches_expected(
        "Computing blue shells: 11 times 7 gives 8.", expected
    )
    assert not operation_matches_expected(
        "Computing red shells: 11 times 7 gives 9.", expected
    )


def test_final_answer_and_achieved_state_are_conservative():
    record = {
        "reasoning_operations": [
            "so the number of red shells is 11 times 7 = 8 ."
        ],
        "canonical_state_ids": [3, 5, 5],
        "answer": 8,
    }
    assert achieved_next_state_id(
        "Computing red shells: 11 times 7 gives 8.", record, 0
    ) == 5
    assert achieved_next_state_id("red shells equals 8", record, 0) == -1
    assert final_answer_matches("Final answer: \\boxed{8}\n", 8)
    assert achieved_next_state_id("Final answer: \\boxed{8}\n", record, 1) == 5

