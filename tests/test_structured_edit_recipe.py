from functools import partial
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    MASK_TOKEN,
    faithful_token_edit_vocab,
)
from textjepa.data.sampling import GroupedTrajectoryBatchSampler
from textjepa.data.token_edit_distance import (
    boundary_token_edit_distance, exact_one_step_advantage,
)
from textjepa.models.edit_jepa import EditJEPA
from textjepa.models.ema import EMATeacher
from textjepa.models.predictor import TokenAlignedEditPredictor
from textjepa.objectives import (
    GoalAdvantageDistill, RefinementActionPrior,
    TokenAlignedCounterfactualPrediction,
)
from scripts.audit_faithful_token_edits import shuffled_action_prediction


def test_gar_pairwise_loss_rewards_correct_same_state_ordering():
    target = torch.tensor([[0.2]])
    alt_target = torch.tensor([[[0.0, -0.1]]])
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    common = {
        "gar_action_target": target,
        "gar_alt_action_target": alt_target,
        "gar_alt_action_valid": valid,
    }
    correct = SimpleNamespace(
        preds=torch.zeros(1), step_mask=torch.ones(1, 1, dtype=torch.bool),
        extras={**common, "gar_action_value": torch.tensor([[0.3]]),
                "gar_alt_action_value": torch.tensor([[[0.0, -0.2]]])},
    )
    reversed_order = SimpleNamespace(
        preds=torch.zeros(1), step_mask=correct.step_mask,
        extras={**common, "gar_action_value": torch.tensor([[-0.3]]),
                "gar_alt_action_value": torch.tensor([[[0.0, 0.2]]])},
    )
    objective = GoalAdvantageDistill(
        regression_weight=0.0, pairwise_weight=1.0,
        margin=0.1, label_gap=0.001,
    )
    assert objective(correct, {}) < objective(reversed_order, {})


def test_exact_token_edit_teacher_preserves_boundaries():
    assert boundary_token_edit_distance([[1], [2]], [[1, 2]]) == 1
    assert exact_one_step_advantage(
        [[1, 9], [2]], [[1], [2]], [[1], [2]]
    ) == 1


def test_exact_token_edit_gar_targets_expert_and_counterfactual_actions():
    vocab = faithful_token_edit_vocab()
    default_item = FaithfulTokenEditDataset(
        vocab, size=1, seed=39, min_edits=4, max_edits=4,
    )[0]
    assert "gar_token_edit_target" not in default_item
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=39, min_edits=4, max_edits=4,
        corruption_mode="mixed", counterfactual_k=3,
        counterfactual_source="deployable_mixed",
        gar_teacher="token_edit_distance",
    )
    item = dataset[0]
    target = item["buffers"][-1]
    assert item["gar_token_edit_target"] == [
        exact_one_step_advantage(before, after, target)
        for before, after in zip(item["buffers"], item["buffers"][1:])
    ]
    assert item["gar_alt_token_edit_target"] == [
        [exact_one_step_advantage(before, outcome, target) for outcome in outcomes]
        for before, outcomes in zip(item["buffers"], item["alt_buffers"])
    ]
    batch = collate_edits([dataset[0], dataset[1]], vocab.pad_id)
    assert batch["gar_token_edit_target"].shape == batch["step_mask"].shape
    assert batch["gar_alt_token_edit_target"].shape == batch["alt_valid"].shape

    out = SimpleNamespace(
        preds=torch.zeros(1), step_mask=batch["step_mask"], extras={
            "gar_action_value": torch.zeros_like(
                batch["gar_token_edit_target"], dtype=torch.float
            ),
            "gar_action_target": torch.full_like(
                batch["gar_token_edit_target"], 999, dtype=torch.float
            ),
            "gar_alt_action_value": torch.zeros_like(
                batch["gar_alt_token_edit_target"], dtype=torch.float
            ),
            "gar_alt_action_target": torch.full_like(
                batch["gar_alt_token_edit_target"], 999, dtype=torch.float
            ),
            "gar_alt_action_valid": batch["alt_valid"],
        },
    )
    loss = GoalAdvantageDistill(teacher="token_edit_distance")(out, batch)
    assert torch.isfinite(loss)
    assert loss < 2.0  # synthetic batch target overrides latent target 999


def _dataset(mode: str):
    vocab = faithful_token_edit_vocab()
    return vocab, FaithfulTokenEditDataset(
        vocab, size=4, seed=41, min_edits=3, max_edits=4,
        corruption_mode=mode,
    )


def test_ema_teacher_stays_in_eval_mode():
    teacher = EMATeacher(torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.Dropout(0.9)
    ))
    teacher.train(True)
    assert not teacher.training
    assert not teacher.module.training


def test_corruption_modes_are_distinct_and_exact():
    vocab, masked = _dataset("mask")
    item = masked[0]
    assert any(vocab.token_to_id[MASK_TOKEN] in step for step in item["buffers"][0])
    assert set(item["op"]) == {2}
    assert item["buffers"][-1] == masked.source[0]["steps"]

    _, replaced = _dataset("replace")
    item = replaced[0]
    assert set(item["op"]) == {2}
    assert item["resolved_n"][0] == item["resolved_n"][-1]
    assert item["buffers"][-1] == replaced.source[0]["steps"]

    _, removed = _dataset("remove")
    item = removed[0]
    assert set(item["op"]) == {1}
    assert item["resolved_n"][0] < item["resolved_n"][-1]
    assert item["buffers"][-1] == removed.source[0]["steps"]


def test_iterative_refinement_is_fully_masked_replace_only_and_diverse():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=47, max_op=6, max_edge=12,
        op_range=(3, 6), corruption_mode="iterative_refinement",
        trajectory_variants=4, refinement_probability=1.0,
        gar_teacher="token_edit_distance", proposal_pool_k=4,
        proposal_token_pool="prompt_plus_current",
    )
    mask = vocab.token_to_id[MASK_TOKEN]
    variants = [dataset[index] for index in range(4)]
    assert len(dataset) == 8
    assert all(
        all(token == mask for sentence in item["buffers"][0] for token in sentence)
        for item in variants
    )
    assert all(set(item["op"]) == {2} for item in variants)
    assert all(
        all(set(step) == {2} for step in item["proposal_op"])
        for item in variants
    )
    assert all(item["buffers"][-1] == dataset.source[0]["steps"] for item in variants)
    assert len({tuple(map(tuple, item["buffers"][1])) for item in variants}) > 1
    assert any(advantage == 0 for advantage in variants[0]["gar_token_edit_target"])
    assert variants[0]["trajectory_variant"] == 0
    assert variants[3]["trajectory_variant"] == 3


def test_grouped_trajectory_sampler_emits_n_by_m_batches():
    sampler = GroupedTrajectoryBatchSampler(
        base_size=6, variants=4, bases_per_batch=2, seed=3,
        fresh_per_epoch=True,
    )
    batches = list(sampler)
    assert len(batches) == 3
    assert all(len(batch) == 8 for batch in batches)
    for batch in batches:
        groups = [batch[offset:offset + 4] for offset in range(0, 8, 4)]
        assert all(group == list(range(group[0], group[0] + 4)) for group in groups)
    sampler.set_epoch(1)
    assert min(index // 4 for batch in sampler for index in batch) >= 6

    micro = GroupedTrajectoryBatchSampler(
        base_size=4, variants=4, bases_per_batch=2, seed=3,
        fresh_per_epoch=False, microbatch_size=2,
    )
    chunks = list(micro)
    assert len(chunks) == 8
    assert all(len(chunk) == 2 for chunk in chunks)
    assert chunks[0] + chunks[1] == list(range(chunks[0][0], chunks[0][0] + 4))
    assert chunks[2] + chunks[3] == list(range(chunks[2][0], chunks[2][0] + 4))


def test_refinement_prior_supervises_pointer_and_full_vocab_content():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=51, max_op=5, max_edge=10,
        op_range=(3, 5), corruption_mode="iterative_refinement",
        trajectory_variants=2, refinement_probability=0.5,
    )
    batch = collate_edits([dataset[0], dataset[1]], vocab.pad_id)
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        predictor_layers=1, predictor_heads=4, max_chunk_len=320,
        max_buffer_len=16, d_action=8, macro_k=0, token_aligned=True,
        token_predictor_layers=1, dropout=0.0, refinement_prior=True,
    )
    out = model(batch)
    assert out.extras["refinement_position_logits"].shape[:2] == out.step_mask.shape
    assert out.extras["refinement_content_logits"].shape == (
        *out.step_mask.shape, len(vocab)
    )
    loss = RefinementActionPrior()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.refinement_content_head[-1].weight.grad is not None


def test_curriculum_changes_corruption_family_by_epoch():
    _, dataset = _dataset("curriculum")
    dataset.curriculum_epochs = 4
    dataset.set_epoch(0)
    assert dataset._active_corruption_mode() == "mask"
    dataset.set_epoch(1)
    assert dataset._active_corruption_mode() == "replace"
    dataset.set_epoch(2)
    assert dataset._active_corruption_mode() == "mixed"


def test_fresh_per_epoch_is_explicit_and_reproducible():
    vocab = faithful_token_edit_vocab()
    common = dict(
        vocab=vocab, size=4, seed=43, min_edits=8, max_edits=8,
        corruption_mode="mixed",
    )
    fixed = FaithfulTokenEditDataset(**common, fresh_per_epoch=False)
    before = [fixed[index]["buffers"][0] for index in range(len(fixed))]
    fixed.set_epoch(1)
    assert [fixed[index]["buffers"][0] for index in range(len(fixed))] == before

    fresh = FaithfulTokenEditDataset(**common, fresh_per_epoch=True)
    before = [fresh[index]["buffers"][0] for index in range(len(fresh))]
    fresh.set_epoch(1)
    after = [fresh[index]["buffers"][0] for index in range(len(fresh))]
    assert after != before
    repeat = FaithfulTokenEditDataset(**common, fresh_per_epoch=True)
    repeat.set_epoch(1)
    assert [repeat[index]["buffers"][0] for index in range(len(repeat))] == after


def test_pointer_action_is_invariant_to_prefix_shift():
    torch.manual_seed(0)
    predictor = TokenAlignedEditPredictor(16, 8, n_layers=1, n_heads=4)
    x, y, z, prefix, content = torch.randn(5, 16)
    state_a = torch.stack([x, y, z]).unsqueeze(0)
    state_b = torch.stack([prefix, x, y, z]).unsqueeze(0)
    code_a = predictor.encode_action(
        state_a, torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([1]), torch.tensor([2]), content.unsqueeze(0),
    )
    code_b = predictor.encode_action(
        state_b, torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([1]), torch.tensor([3]), content.unsqueeze(0),
    )
    torch.testing.assert_close(code_a, code_b)


def test_structured_model_forward_and_recursive_shapes():
    vocab, dataset = _dataset("mask")
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        predictor_layers=1, predictor_heads=4, max_chunk_len=320,
        max_buffer_len=16, d_action=8, macro_k=0, token_aligned=True,
        token_predictor_layers=1, dropout=0.0,
    )
    out = model(batch)
    assert out.extras["token_predictions"].shape[:2] == out.step_mask.shape
    assert out.extras["token_rollout_predictions"].shape == out.extras["token_predictions"].shape
    assert out.actions.shape[-1] == 8
    assert torch.isfinite(out.preds).all()
    assert torch.isfinite(out.rollout).all()
    assert out.extras["gar_action_value"].shape == out.step_mask.shape
    assert not out.extras["gar_action_target"].requires_grad
    assert torch.isfinite(GoalAdvantageDistill()(out, batch))

    shuffled, reason, changed = shuffled_action_prediction(
        model, out, batch, return_output=True
    )
    assert reason is None
    assert bool(changed.any())
    assert shuffled.extras["token_predictions"].shape == out.extras[
        "token_predictions"
    ].shape


def test_structured_counterfactuals_supervise_exact_token_outcomes():
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=47, min_edits=3, max_edits=4,
        corruption_mode="mixed", counterfactual_k=2,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    assert batch["alt_op"].shape == batch["alt_valid"].shape
    assert batch["alt_edit_position"].shape == batch["alt_valid"].shape
    assert batch["alt_edit_content_token"].shape == batch["alt_valid"].shape
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        predictor_layers=1, predictor_heads=4, max_chunk_len=320,
        max_buffer_len=16, d_action=8, macro_k=0, token_aligned=True,
        token_predictor_layers=1, dropout=0.0,
    )
    B, T, K = batch["alt_op"].shape
    C, L = batch["alt_buffer_tokens"].shape[-2:]
    alternative_buffers = batch["alt_buffer_tokens"].reshape(B, T * K, C, L)
    with torch.no_grad():
        full_states, full_mask = model.encode_token_buffers(
            alternative_buffers, mode="teacher"
        )
        model.counterfactual_encode_chunk_states = 2
        chunked_states, chunked_mask = model.encode_token_buffers_chunked(
            alternative_buffers, mode="teacher"
        )
    assert torch.equal(full_mask, chunked_mask)
    torch.testing.assert_close(full_states, chunked_states)
    out = model(batch)
    assert out.extras["cf_token_pred"].shape == out.extras["cf_token_tgt"].shape
    assert out.extras["cf_token_valid"].shape == batch["alt_valid"].shape
    loss = TokenAlignedCounterfactualPrediction()(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.token_pred.out.weight.grad is not None
    assert out.extras["gar_alt_action_value"].shape == batch["alt_valid"].shape
    assert out.extras["gar_alt_action_target"].shape == batch["alt_valid"].shape
    assert not out.extras["gar_alt_action_target"].requires_grad
    model.zero_grad(set_to_none=True)
    out = model(batch)
    gar_loss = GoalAdvantageDistill()(out, batch)
    gar_loss.backward()
    assert torch.isfinite(gar_loss)
    assert model.gar_head[-1].weight.grad is not None
