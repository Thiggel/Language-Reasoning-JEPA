from functools import partial

import torch
from torch.utils.data import DataLoader

from scripts.audit_faithful_token_edits import shuffled_action_prediction
from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    OPS,
    _apply,
    _counterfactual_exclusions,
    faithful_token_edit_vocab,
)
from textjepa.models.edit_jepa import EditJEPA
from textjepa.objectives.counterfactual import (
    CounterfactualOutcomePrediction,
    CounterfactualSlotPrediction,
)
from textjepa.objectives.chunk_pred import SlotAnchor
from textjepa.objectives.delta_action import ObservedActionLDAD


def test_faithful_token_edits_are_text_only_and_recover_target():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=91, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=3, max_edits=4,
    )
    for index in range(2):
        item = dataset[index]
        assert len(item["actions"]) >= 3
        assert item["remaining"][-1] == 0
        action_text = vocab.decode(item["actions"][0])
        assert "token position" in action_text
        assert not any(word in action_text for word in ("ancestor", "necessary"))
        assert "target_tokens" not in item
        assert item["buffers"][-1] == dataset.source[index]["steps"]
        assert all(item["changed"])
        assert all(mask == [] for mask in item["defect_masks"])


def test_faithful_token_edits_preserve_official_multistep_segmentation():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=4, seed=97, max_op=10, max_edge=16,
        op_range=(6, 10), min_edits=8, max_edits=8,
    )
    for index in range(4):
        source_steps = dataset.source[index]["steps"]
        item = dataset[index]
        assert len(source_steps) > 1
        assert len(item["buffers"][-1]) == len(source_steps)
        assert item["buffers"][-1] == source_steps
        # Official iGSM usually ends steps with fused numeric punctuation,
        # not the standalone period token used by the old segmentation.
        assert any(sentence[-1] != vocab.token_to_id["."] for sentence in source_steps)


def test_boundary_edits_are_literal_and_causally_invertible():
    # At the flattened position between steps, insertion belongs to the step
    # on the right.  Deleting that first token and inserting it back therefore
    # recovers both tokens and structure without a hidden boundary label.
    buffer = [[10, 11], [20, 21]]
    _apply(buffer, ("insert", 2, 19))
    assert buffer == [[10, 11], [19, 20, 21]]
    _apply(buffer, ("delete", 2, None))
    assert buffer == [[10, 11], [20, 21]]
    _apply(buffer, ("delete", 2, None))
    assert buffer == [[10, 11], [21]]
    _apply(buffer, ("insert", 2, 20))
    assert buffer == [[10, 11], [20, 21]]


def test_every_recorded_edit_is_a_literal_flat_token_transition():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=3, seed=101, max_op=8, max_edge=14,
        op_range=(5, 8), min_edits=10, max_edits=10,
    )
    for dataset_index in range(len(dataset)):
        item = dataset[dataset_index]
        positions = []
        for action in item["actions"]:
            words = vocab.decode(action).split()
            positions.append(int(words[words.index("position") + 1]))
        for before_buffer, after_buffer, op, position in zip(
            item["buffers"], item["buffers"][1:], item["op"], positions
        ):
            before = [token for sentence in before_buffer for token in sentence]
            after = [token for sentence in after_buffer for token in sentence]
            if op == 0:
                assert after == before[:position] + before[position + 1:]
            elif op == 1:
                assert before == after[:position] + after[position + 1:]
            else:
                assert len(after) == len(before)
                assert before[:position] == after[:position]
                assert before[position + 1:] == after[position + 1:]


def _decoded_action(vocab, encoded):
    words = vocab.decode(encoded).split()
    kind = words[0]
    position = int(words[words.index("position") + 1])
    token = None
    if "with" in words:
        token = vocab.token_to_id[words[words.index("with") + 1]]
    return kind, position, token


def test_counterfactual_outcomes_execute_exactly_without_quality_labels():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=107, max_op=8, max_edge=14,
        op_range=(5, 8), min_edits=6, max_edits=6,
        counterfactual_k=5, counterfactual_source="mixed",
    )
    item = dataset[0]
    assert "alt_remaining" not in item
    assert "alt_defects" not in item
    assert len(item["alt_changed"]) == len(item["alt_actions"])
    for before, expert, actions, outcomes in zip(
        item["buffers"], item["actions"], item["alt_actions"],
        item["alt_buffers"]
    ):
        observed_pool = {token for sentence in before for token in sentence}
        assert {vocab.decode(action).split()[0] for action in actions} >= {
            "insert", "delete", "replace"
        }
        for encoded, expected in zip(actions, outcomes):
            action = _decoded_action(vocab, encoded)
            assert action != _decoded_action(vocab, expert)
            if action[2] is not None:
                assert action[2] in observed_pool
            actual = [list(sentence) for sentence in before]
            _apply(actual, action)
            assert actual == expected


def test_counterfactuals_are_deterministic_prefixes_and_do_not_change_expert():
    vocab = faithful_token_edit_vocab()
    common = dict(
        vocab=vocab, size=1, seed=109, max_op=8, max_edge=14,
        op_range=(5, 8), min_edits=6, max_edits=6,
        counterfactual_source="uniform_local",
    )
    zero = FaithfulTokenEditDataset(**common, counterfactual_k=0)[0]
    two = FaithfulTokenEditDataset(**common, counterfactual_k=2)[0]
    five = FaithfulTokenEditDataset(**common, counterfactual_k=5)[0]
    repeat = FaithfulTokenEditDataset(**common, counterfactual_k=5)[0]
    assert "alt_actions" not in zero and "alt_buffers" not in zero
    for key in zero:
        assert two[key] == zero[key] == five[key]
    assert five["alt_actions"] == repeat["alt_actions"]
    assert five["alt_buffers"] == repeat["alt_buffers"]
    assert five["alt_changed"] == repeat["alt_changed"]
    assert [step[:2] for step in five["alt_actions"]] == two["alt_actions"]
    assert [step[:2] for step in five["alt_buffers"]] == two["alt_buffers"]
    assert [step[:2] for step in five["alt_changed"]] == two["alt_changed"]


def test_deployable_mixed_is_prefix_stable_and_balances_k2_operations():
    vocab = faithful_token_edit_vocab()
    common = dict(
        vocab=vocab, size=4, seed=127, max_op=8, max_edge=14,
        op_range=(5, 8), min_edits=6, max_edits=6,
        counterfactual_source="deployable_mixed",
    )
    k2 = FaithfulTokenEditDataset(**common, counterfactual_k=2)
    k4 = FaithfulTokenEditDataset(**common, counterfactual_k=4)
    expert = ("replace", 3, 17)
    assert _counterfactual_exclusions("deployable_mixed", expert) == set()
    assert _counterfactual_exclusions("mixed", expert) == {expert}
    counts = {op: 0 for op in OPS.values()}
    for index in range(len(k2)):
        two, four = k2[index], k4[index]
        assert two["buffers"] == four["buffers"]
        assert two["actions"] == four["actions"]
        assert [step[:2] for step in four["alt_actions"]] == two["alt_actions"]
        assert [step[:2] for step in four["alt_buffers"]] == two["alt_buffers"]
        for step in two["alt_op"]:
            for operation in step:
                counts[operation] += 1
    assert all(counts.values())
    assert len(set(counts.values())) == 1


def test_counterfactual_collate_pads_actions_and_nested_buffer_outcomes():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=113, max_op=8, max_edge=14,
        op_range=(5, 8), min_edits=5, max_edits=7,
        counterfactual_k=3, counterfactual_source="mixed",
    )
    batch = collate_edits([dataset[0], dataset[1]], vocab.pad_id)
    assert batch["alt_tokens"].shape[:3] == (2, batch["step_mask"].shape[1], 3)
    assert batch["alt_buffer_tokens"].shape[:3] == batch["alt_tokens"].shape[:3]
    assert batch["alt_buffer_mask"].shape[:3] == batch["alt_tokens"].shape[:3]
    assert batch["alt_valid"].shape == batch["alt_tokens"].shape[:3]
    assert batch["alt_changed_tokens"].shape[:3] == batch["alt_tokens"].shape[:3]
    assert torch.equal(batch["alt_changed_valid"], batch["alt_valid"])
    assert batch["alt_valid"].sum() == sum(
        len(step) for index in range(2) for step in dataset[index]["alt_actions"]
    )


def test_counterfactual_outcomes_supervise_dynamics_without_quality_labels():
    torch.manual_seed(7)
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=127, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=4, max_edits=4,
        counterfactual_k=2, counterfactual_source="mixed",
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=128, d_action=8, predictor_layers=1,
        predictor_heads=4, macro_k=0, chunk_target="frozen",
    )
    out = model(batch)
    assert out.extras["cf_chunk_pred"].shape == out.extras["cf_chunk_tgt"].shape
    assert out.extras["cf_valid"].shape == batch["alt_valid"].shape
    assert out.extras["cf_slot_pred"].shape == out.extras["cf_slot_tgt"].shape
    assert "alt_remaining" not in batch
    loss = (
        CounterfactualOutcomePrediction()(out, batch)
        + CounterfactualSlotPrediction()(out, batch)
        + SlotAnchor()(out, batch)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.predictor.inp.weight.grad is not None


def test_attention_buffer_predictor_preserves_local_counterfactual_outputs():
    torch.manual_seed(17)
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=131, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=4, max_edits=4,
        counterfactual_k=2, counterfactual_source="mixed",
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=128, d_action=8, predictor_layers=1,
        predictor_heads=4, macro_k=0, chunk_target="frozen",
        attn_predictor=True,
    )
    out = model(batch)
    assert out.extras["cf_slot_pred"].shape == out.extras["cf_slot_tgt"].shape
    shuffled, reason, changed = shuffled_action_prediction(model, out, batch)
    assert reason is None
    assert shuffled.shape == out.preds.shape
    assert bool(changed.any())
    loss = CounterfactualSlotPrediction()(out, batch) + SlotAnchor()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.attn_pred.a_proj.weight.grad is not None


def test_faithful_token_edit_model_is_causal_hierarchical_and_ldad_trains():
    torch.manual_seed(4)
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=93, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=4, max_edits=4,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=128,
        d_action=8, d_macro=4, macro_k=2, predictor_layers=1,
        predictor_heads=4, observed_action_ldad=True,
        dense_rollout_depth=2, high_dense_rollout_depth=2,
    )
    out = model(batch)
    assert model.core.predictor.causal_sequence
    assert model.core.hi_predictor.causal_sequence
    assert out.hi_preds is not None
    assert "dense_rollout_predictions" in out.extras
    assert "high_dense_rollout_predictions" in out.extras
    loss = ObservedActionLDAD()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.chunk_encoder.tok.weight.grad is not None


def test_long_faithful_buffers_remain_multiple_bounded_steps():
    """Long full/hard buffers must not collapse into one 276-token chunk."""
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=8, seed=103, max_op=21, max_edge=28,
        op_range=(16, 21), min_edits=6, max_edits=6,
    )
    item = max(
        (dataset[index] for index in range(len(dataset))),
        key=lambda x: sum(map(len, x["buffers"][-1])),
    )
    terminal = item["buffers"][-1]
    total_length = sum(map(len, terminal))
    assert total_length > 128
    assert len(terminal) > 1
    assert max(map(len, terminal)) < total_length

    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=max(map(len, terminal)), d_action=8, predictor_layers=1,
        predictor_heads=4,
    )
    padded = torch.full(
        (1, len(terminal), max(map(len, terminal))), vocab.pad_id,
        dtype=torch.long,
    )
    for index, sentence in enumerate(terminal):
        padded[0, index, :len(sentence)] = torch.tensor(sentence)
    encoded = model.encode_chunks(padded)
    assert encoded.shape == (1, len(terminal), 32)
