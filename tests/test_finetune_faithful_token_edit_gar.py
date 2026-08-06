from scripts.finetune_faithful_token_edit_gar import candidate_labels


def test_candidate_labels_are_exact_edit_distance_advantages():
    current = [[1, 2, 3]]
    target = [[1, 4, 3]]
    candidates = [("replace", 1, 4), ("delete", 1, None), ("insert", 1, 4)]
    assert candidate_labels(current, target, candidates) == [1.0, 0.0, 0.0]
