from functools import partial

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from scripts.train_sentence_hierarchy import compute_sentence_losses
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.sentence_hierarchy import SentenceHierarchyJEPA


def batch_and_vocab(size=3):
    vocab = build_vocab(23)
    dataset = SemanticBoundaryLMDataset(
        vocab, size=size, seed=17, boundary_mode="semantic", modulus=23,
        n_vars_range=(8, 10), leaf_prob=0.35, steps_range=(4, 6),
        distractor_prob=0.0, max_distractors=0,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=size,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )))
    return batch, vocab


def tiny_model(vocab, d_macro=8):
    return SentenceHierarchyJEPA(
        len(vocab), vocab.pad_id, d_low=24, d_high=16,
        encoder_layers=1, high_encoder_layers=1, predictor_layers=1,
        n_heads=4, ff_mult=2, max_len=768, d_token_action=8,
        d_macro=d_macro, macro_layers=2, low_dense_depth=2,
        high_dense_depth=2, use_token_prior=True,
    )


def test_sentence_states_are_separate_and_taken_after_complete_sentence():
    torch.manual_seed(2)
    batch, vocab = batch_and_vocab(2)
    model = tiny_model(vocab).eval()
    with torch.no_grad():
        out = model(
            batch["tokens"], batch["prompt_len"], batch["sentence_ends"]
        )
    level = out["sentence_level"]
    assert out["states"].shape[-1] == 24
    assert out["high_states"].shape[-1] == 16
    assert model.encoder is not model.high_encoder
    assert not any(
        left.data_ptr() == right.data_ptr()
        for left in model.encoder.parameters()
        for right in model.high_encoder.parameters()
    )
    for row in range(2):
        prompt = int(batch["prompt_len"][row])
        ends = batch["sentence_ends"][row]
        ends = ends[ends > 0]
        for column, end_tensor in enumerate(ends):
            end = int(end_tensor)
            source_position = prompt - 1 if column == 0 else prompt + int(ends[column - 1]) - 1
            target_position = prompt + end - 1
            assert level["source_positions"][row, column] == source_position
            assert level["target_positions"][row, column] == target_position
            assert torch.allclose(
                level["prev"][row, column], out["high_states"][row, source_position]
            )
            assert torch.allclose(
                level["target"][row, column], out["high_targets"][row, target_position]
            )


def test_sentence_state_and_code_are_causal_at_the_completed_boundary():
    torch.manual_seed(3)
    batch, vocab = batch_and_vocab(2)
    model = tiny_model(vocab).eval()
    with torch.no_grad():
        original = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    changed = {name: value.clone() for name, value in batch.items()}
    prompt = int(batch["prompt_len"][0])
    first_end = int(batch["sentence_ends"][0, 0])
    changed["tokens"][0, prompt + first_end:] = torch.randint(
        1, len(vocab), changed["tokens"][0, prompt + first_end:].shape
    )
    with torch.no_grad():
        other = model(changed["tokens"], changed["prompt_len"], changed["sentence_ends"])
    left, right = original["sentence_level"], other["sentence_level"]
    assert torch.allclose(left["target"][0, 0], right["target"][0, 0])
    assert torch.allclose(left["codes"][0, 0], right["codes"][0, 0])


def test_bidirectional_cls_macro_encoder_accepts_variable_sentence_lengths():
    torch.manual_seed(5)
    batch, vocab = batch_and_vocab(3)
    model = tiny_model(vocab, d_macro=7).eval()
    with torch.no_grad():
        level = model(
            batch["tokens"], batch["prompt_len"], batch["sentence_ends"]
        )["sentence_level"]
    assert level["codes"].shape[-1] == 7
    lengths = level["raw_action_valid"].sum(-1)[level["valid"]]
    assert lengths.unique().numel() > 1
    assert torch.isfinite(level["codes"][level["valid"]]).all()


def test_high_predictor_is_causal_over_sentence_history_only():
    torch.manual_seed(7)
    batch, vocab = batch_and_vocab(2)
    model = tiny_model(vocab).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    level = out["sentence_level"]
    assert model.high_predictor.causal_sequence
    changed_codes = level["codes"].clone()
    changed_codes[:, 1:] = torch.randn_like(changed_codes[:, 1:]) * 20
    with torch.no_grad():
        changed = model.high_predictor(level["prev"], changed_codes, level["valid"])
    assert torch.allclose(level["pred"][:, 0], changed[:, 0], atol=1e-6)


def test_sentence_objectives_reach_both_encoders_and_high_predictor():
    torch.manual_seed(11)
    batch, vocab = batch_and_vocab(3)
    model = tiny_model(vocab).train()
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    loss = (
        (out["low_pred"] - out["target"].detach()).square()[out["valid"]].mean()
        + (out["sentence_level"]["pred"] - out["sentence_level"]["target"].detach())
        .square()[out["sentence_level"]["valid"]].mean()
        + (out["sentence_level"]["low_endpoint_high"]
           - out["sentence_level"]["target"].detach())
        .square()[out["sentence_level"]["valid"]].mean()
    )
    loss.backward()
    for module in (model.encoder, model.high_encoder, model.low_predictor,
                   model.high_predictor, model.low_to_high):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_counterfactual_gar_uses_real_appended_sentences_without_goal_input():
    torch.manual_seed(13)
    batch, vocab = batch_and_vocab(3)
    model = tiny_model(vocab).train()
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    counterfactual = model.sentence_counterfactuals(
        out, batch["tokens"], batch["prompt_len"], k=3,
    )
    n = int(out["sentence_level"]["valid"].sum())
    assert counterfactual["value"].shape == (n, 3)
    assert counterfactual["advantage_target"].shape == (n, 3)
    assert counterfactual["predicted_outcome"].shape[-1] == model.d_high
    # Candidate zero is the factual sentence and must reproduce the teacher
    # target selected at the completed-sentence boundary.
    anchors = counterfactual["anchor_indices"]
    factual_target = out["sentence_level"]["target"][anchors[:, 0], anchors[:, 1]]
    assert torch.allclose(
        counterfactual["exact_outcome"][:, 0], factual_target, atol=1e-5
    )
    # The deployed value head has no terminal-goal argument by construction.
    assert model.macro_value.net[0].normalized_shape == (
        2 * model.d_high + model.d_macro,
    )


def test_nearest_counterfactuals_and_joint_mse_ranking_backpropagate():
    torch.manual_seed(19)
    batch, vocab = batch_and_vocab(3)
    model = tiny_model(vocab).train()
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    cf = model.sentence_counterfactuals(
        out, batch["tokens"], batch["prompt_len"], k=4, source="nearest"
    )
    mse = torch.nn.functional.smooth_l1_loss(cf["value"], cf["advantage_target"])
    dynamics = torch.nn.functional.mse_loss(
        cf["predicted_outcome"], cf["exact_outcome"].detach()
    )
    better = cf["advantage_target"].unsqueeze(2) > cf["advantage_target"].unsqueeze(1)
    pair = torch.relu(0.1 - cf["value"].unsqueeze(2) + cf["value"].unsqueeze(1))
    ranking = pair[better].mean() if better.any() else pair.sum() * 0
    (mse + dynamics + ranking).backward()
    for module in (model.high_predictor, model.macro_value, model.macro_action.encoder):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_full_sentence_training_loss_is_finite_and_reports_all_components():
    torch.manual_seed(23)
    batch, vocab = batch_and_vocab(3)
    model = tiny_model(vocab).train()
    cfg = OmegaConf.load("configs/sentence_hierarchy.yaml")
    cfg.objective.gar_weight = 1.0
    cfg.objective.gar_k = 3
    cfg.objective.temporal_straightening = 0.1
    cfg.objective.value_monotonicity = 0.1
    cfg.objective.macro_prior = 0.05
    cfg.objective.support = 0.1
    cfg.objective.bridge = 0.25
    cfg.objective.transition_reachability = 0.25
    cfg.objective.reachability_classifier = 0.1
    out = model(batch["tokens"], batch["prompt_len"], batch["sentence_ends"])
    loss, items = compute_sentence_losses(out, cfg, model, batch)
    assert torch.isfinite(loss)
    expected = {
        "low_prediction", "token_prior", "high_prediction", "high_dense",
        "bridge", "transition_reachability", "reachability_classifier",
        "temporal_straightening", "value_monotonicity", "gar_regression",
        "gar_ranking", "gar_counterfactual_mse",
    }
    assert expected <= items.keys()
    assert items["macro_prior"] >= 0
    loss.backward()
    for module in (
        model.high_encoder, model.high_predictor, model.macro_action,
        model.macro_value, model.reachability, model.planning_projection,
    ):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_shared_space_control_is_explicit_and_dimension_matched():
    _, vocab = batch_and_vocab(1)
    shared = SentenceHierarchyJEPA(
        len(vocab), vocab.pad_id, d_low=24, d_high=24,
        encoder_layers=1, high_encoder_layers=1, predictor_layers=1,
        n_heads=4, ff_mult=2, d_token_action=8, d_macro=6,
        macro_hidden=16, macro_heads=4, separate_high_encoder=False,
    )
    assert shared.high_encoder is shared.encoder
    assert shared.high_teacher is shared.teacher
    try:
        SentenceHierarchyJEPA(
            len(vocab), vocab.pad_id, d_low=24, d_high=16,
            encoder_layers=1, high_encoder_layers=1, predictor_layers=1,
            n_heads=4, ff_mult=2, d_token_action=8, d_macro=6,
            macro_hidden=16, macro_heads=4, separate_high_encoder=False,
        )
    except ValueError as error:
        assert "matching state widths" in str(error)
    else:
        raise AssertionError("a shared state space cannot silently project dimensions")
