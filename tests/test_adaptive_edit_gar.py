from functools import partial
from pathlib import Path

from hydra import compose, initialize_config_dir
import torch
from torch.utils.data import DataLoader

from textjepa.data.edits.dataset import collate_edits
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    _proposal_tokens,
    faithful_token_edit_vocab,
)
from textjepa.models.edit_jepa import EditJEPA
from textjepa.objectives.prediction import TokenAlignedCounterfactualPrediction
from textjepa.objectives.value import GoalAdvantageDistill


def small_dataset(**overrides):
    vocab = faithful_token_edit_vocab()
    kwargs = dict(
        vocab=vocab, size=2, seed=211, max_op=6, max_edge=12,
        op_range=(3, 6), min_edits=4, max_edits=4,
        proposal_pool_k=4,
    )
    kwargs.update(overrides)
    return vocab, FaithfulTokenEditDataset(**kwargs)


def test_adaptive_selection_takes_highest_valid_predicted_scores():
    scores = torch.tensor([[[0.1, 8.0, 3.0, 2.0], [5.0, 4.0, 9.0, 1.0]]])
    valid = torch.tensor([[[True, False, True, True], [True, True, False, True]]])
    indices, selected_valid = EditJEPA.select_adaptive_candidates(
        scores, valid, selected_k=2
    )
    assert indices.tolist() == [[[2, 3], [0, 1]]]
    assert selected_valid.all()


def test_random_selection_is_deterministic_and_independent_of_gar_values():
    example = torch.tensor([17, 23])
    first = EditJEPA.proposal_ranking_scores(
        torch.randn(2, 3, 7), "random", example
    )
    second = EditJEPA.proposal_ranking_scores(
        torch.randn(2, 3, 7) * 100, "random", example
    )
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    valid = torch.ones_like(first, dtype=torch.bool)
    assert torch.equal(
        EditJEPA.select_adaptive_candidates(first, valid, 3)[0],
        EditJEPA.select_adaptive_candidates(second, valid, 3)[0],
    )


def test_proposal_token_pools_match_planner_information_regimes():
    prompt = [[9, 2, 8]]
    current = [[1, 2], [3, 1]]
    assert _proposal_tokens(prompt, current, "current_buffer") == [1, 2, 3]
    assert _proposal_tokens(prompt, current, "prompt_plus_current") == [
        9, 2, 8, 1, 3
    ]


def test_broad_deployable_pool_is_target_independent_and_current_buffer_only():
    vocab, uniform = small_dataset(counterfactual_source="uniform_local")
    _, mixed = small_dataset(counterfactual_source="mixed")
    first, second = uniform[0], mixed[0]
    # proposal_pool_k forces the same deployable sampler; the legacy
    # counterfactual source (and expert-exclusion behavior) is irrelevant.
    assert first["proposal_actions"] == second["proposal_actions"]
    assert first["proposal_buffers"] == second["proposal_buffers"]
    assert "alt_actions" not in first
    assert "proposal_remaining" not in first
    for before, content_steps in zip(
        first["buffers"], first["proposal_edit_content_token"]
    ):
        observed = {token for sentence in before for token in sentence}
        for token in content_steps:
            if token != vocab.pad_id:
                assert token in observed


def test_adaptive_pool_collates_broad_but_model_executes_selected_shapes_and_grads():
    torch.manual_seed(13)
    vocab, dataset = small_dataset(size=1, proposal_pool_k=64)
    batch = next(iter(DataLoader(
        dataset, batch_size=1,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    assert batch["proposal_op"].shape[-1] == 64
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=128, d_action=8, predictor_layers=1,
        predictor_heads=4, macro_k=0, token_aligned=True,
        token_predictor_layers=1, selected_k=4,
    )
    out = model(batch)
    assert out.extras["adaptive_proposal_scores"].shape == batch["proposal_op"].shape
    assert out.extras["adaptive_selected_indices"].shape == (
        *batch["proposal_op"].shape[:2], 4
    )
    assert out.extras["cf_token_pred"].shape[:3] == (
        *batch["proposal_op"].shape[:2], 4
    )
    assert out.extras["cf_token_pred"].shape == out.extras["cf_token_tgt"].shape
    assert out.extras["gar_alt_action_value"].shape == (
        *batch["proposal_op"].shape[:2], 4
    )
    loss = (
        GoalAdvantageDistill()(out, batch)
        + TokenAlignedCounterfactualPrediction()(out, batch)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.gar_head[-1].weight.grad).all()
    assert torch.isfinite(model.token_pred.out.weight.grad).all()


def test_adaptive_exact_teacher_gathers_selected_k_and_has_finite_gradients():
    torch.manual_seed(19)
    vocab, dataset = small_dataset(
        proposal_pool_k=8, gar_teacher="token_edit_distance"
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=2,
        collate_fn=partial(collate_edits, pad_id=vocab.pad_id),
    )))
    assert batch["gar_proposal_token_edit_target"].shape == batch[
        "proposal_valid"
    ].shape
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        max_chunk_len=128, d_action=8, predictor_layers=1,
        predictor_heads=4, macro_k=0, token_aligned=True,
        token_predictor_layers=1, selected_k=2,
    )
    out = model(batch)
    loss = GoalAdvantageDistill(teacher="token_edit_distance")(out, batch)
    selected = batch["gar_proposal_token_edit_target"].gather(
        -1, out.extras["adaptive_selected_indices"]
    ).to(out.extras["gar_alt_action_value"].dtype)
    assert out.extras[
        "gar_alt_terminal_privileged_token_edit_target"
    ].shape[-1] == 2
    torch.testing.assert_close(
        out.extras["gar_alt_terminal_privileged_token_edit_target"], selected
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.gar_head[-1].weight.grad).all()


def test_adaptive_config_composes_with_bounded_pool_and_selection():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        config_dir=str(root / "configs"), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=edit_token_structured_gar_adaptive_hard"],
        )
    assert cfg.data.counterfactual_k == 0
    assert cfg.data.proposal_pool_k == 64
    assert cfg.data.proposal_token_pool == "current_buffer"
    assert cfg.data.counterfactual_source == "deployable_mixed"
    assert cfg.model.selected_k == 4
    assert cfg.model.proposal_selection == "hard"
    assert cfg.objective.gar_action_value.weight == 1.0
    assert cfg.objective.vicreg.weight == 0.0
    assert cfg.objective.observed_action_ldad.weight == 0.0
    assert not cfg.model.observed_action_ldad

    with initialize_config_dir(
        config_dir=str(root / "configs"), version_base=None
    ):
        exact = compose(
            config_name="config",
            overrides=["+experiment=edit_token_structured_gar_adaptive_exact"],
        )
    assert exact.data.gar_teacher == "token_edit_distance"
    assert exact.data.proposal_pool_k == 16
    assert exact.objective.gar_action_value.teacher == "token_edit_distance"


def test_random_and_prompt_adaptive_configs_are_matched_clean_ema_overrides():
    root = Path(__file__).resolve().parents[1]
    names = {
        "edit_token_structured_gar_adaptive_random": ("random", "current_buffer"),
        "edit_token_structured_gar_adaptive_hard_prompt": (
            "hard", "prompt_plus_current"
        ),
        "edit_token_structured_gar_adaptive_random_prompt": (
            "random", "prompt_plus_current"
        ),
    }
    with initialize_config_dir(
        config_dir=str(root / "configs"), version_base=None
    ):
        for name, (selection, pool) in names.items():
            cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
            assert cfg.data.proposal_pool_k == 64
            assert cfg.model.selected_k == 4
            assert cfg.model.proposal_selection == selection
            assert cfg.data.proposal_token_pool == pool
            assert cfg.objective.vicreg.weight == 0.0
            assert cfg.objective.observed_action_ldad.weight == 0.0
