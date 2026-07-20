from scripts.eval_intent_generalization import variable_range


def test_variable_range_supports_explicit_prompt_size_control():
    assert variable_range(9) == (6, 12)
    assert variable_range(12) == (24, 36)
    assert variable_range(12, (18, 24)) == (18, 24)
