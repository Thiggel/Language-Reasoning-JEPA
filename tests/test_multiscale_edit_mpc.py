import inspect

import torch

from textjepa.data.vocab import Vocab
from textjepa.models.multiscale_edit_jepa import MultiscaleEditJEPA
from textjepa.objectives.refinement import RefinementActionPrior
from textjepa.planning.multiscale_edit_mpc import MultiscaleEditMPC
from textjepa.planning.hierarchical_edit_cem import HierarchicalEditCEM


def _vocab():
    return Vocab(["<mask>", "prompt", "a", "b", "c", "."])


def _model(vocab, detach=True, variant="sentence"):
    return MultiscaleEditJEPA(
        len(vocab), vocab.pad_id, variant, d_model=16, d_action=4,
        d_macro=3, macro_k=2, token_layers=1, sentence_layers=1,
        predictor_layers=1, n_heads=4, max_sequence_len=32,
        max_sentences=4, base_prior=True,
        base_prior_detach_state=detach,
    )


def _batch(vocab):
    mask = vocab.token_to_id["<mask>"]
    a, b = vocab.token_to_id["a"], vocab.token_to_id["b"]
    return {
        "prompt_tokens": torch.tensor([[[vocab.token_to_id["prompt"], 0]]]),
        "prompt_mask": torch.tensor([[True]]),
        "buffer_tokens": torch.tensor([[[[mask, mask]], [[a, mask]], [[a, b]]]]),
        "buffer_mask": torch.tensor([[[True], [True], [True]]]),
        "op": torch.tensor([[2, 2]]),
        "edit_position": torch.tensor([[0, 1]]),
        "edit_content_token": torch.tensor([[a, b]]),
        "step_mask": torch.tensor([[True, True]]),
        "action_tokens": torch.tensor([[[a], [b]]]),
        "gar_token_edit_target": torch.tensor([[1, 1]]),
        "goal_distance": torch.tensor([[2, 1, 0]]),
    }


def test_base_prior_detached_and_attached_are_real_gradient_ablations():
    vocab = _vocab()
    detached = _model(vocab, True)
    RefinementActionPrior()(detached(_batch(vocab)), _batch(vocab)).backward()
    assert detached.base_prior.position[1].weight.grad is not None
    assert detached.encoder.tok.weight.grad is None

    attached = _model(vocab, False)
    RefinementActionPrior()(attached(_batch(vocab)), _batch(vocab)).backward()
    assert attached.encoder.tok.weight.grad is not None
    assert attached.encoder.tok.weight.grad.abs().sum() > 0


def test_planner_api_cannot_receive_clean_target_and_emits_literal_action():
    vocab = _vocab()
    model = _model(vocab).eval()
    planner = MultiscaleEditMPC(
        model, vocab, device="cpu", beam_width=2,
        top_positions=2, top_tokens=2, max_candidates=4,
    )
    assert "target" not in inspect.signature(planner.first_action).parameters
    mask = vocab.token_to_id["<mask>"]
    action, posterior, _ = planner.first_action(
        [[vocab.token_to_id["prompt"]]], [[mask, mask]], horizon=2
    )
    assert action is not None and action[0] == "replace"
    assert action[1] in {0, 1}
    assert action[2] not in planner.excluded
    assert action in posterior


def test_macro_planner_only_scores_codes_built_from_executable_actions():
    vocab = _vocab()
    model = _model(vocab, variant="sentence_macro").eval()
    calls = []
    original = model.macro_model.forward

    def record(actions, *args, **kwargs):
        calls.append(actions.detach().clone())
        return original(actions, *args, **kwargs)

    model.macro_model.forward = record
    planner = MultiscaleEditMPC(
        model, vocab, device="cpu", beam_width=2,
        top_positions=2, top_tokens=2, max_candidates=4,
    )
    mask = vocab.token_to_id["<mask>"]
    planner.first_action(
        [[vocab.token_to_id["prompt"]]], [[mask, mask]], horizon=2
    )
    assert calls
    assert all(value.shape == (1, 2, model.macro_model.encoder.norm.normalized_shape[0])
               for value in calls)


def test_decoder_free_hierarchical_cem_uses_lower_planner_as_inverse():
    vocab = _vocab()
    model = _model(vocab, variant="sentence_macro").eval()
    primitive = MultiscaleEditMPC(
        model, vocab, device="cpu", beam_width=2,
        top_positions=2, top_tokens=2, max_candidates=4,
    )
    planner = HierarchicalEditCEM(
        primitive, mode="subgoal", candidates=4, iterations=1,
        elites=1, reachability_topk=1, low_horizon=2,
    )
    mask = vocab.token_to_id["<mask>"]
    result = planner.first_action(
        [[vocab.token_to_id["prompt"]]], [[mask, mask]]
    )
    assert result.first_action is not None
    assert result.decoded_actions == ()
    assert result.mode == "subgoal"


def test_macro_decoder_cem_returns_an_executable_option():
    vocab = _vocab()
    model = MultiscaleEditJEPA(
        len(vocab), vocab.pad_id, "sentence_macro", d_model=16,
        d_action=4, d_macro=3, macro_k=2, token_layers=1,
        sentence_layers=1, predictor_layers=1, n_heads=4,
        max_sequence_len=32, max_sentences=4, base_prior=True,
        macro_decoder=True,
    ).eval()
    primitive = MultiscaleEditMPC(
        model, vocab, device="cpu", beam_width=2,
        top_positions=2, top_tokens=2, max_candidates=4,
    )
    planner = HierarchicalEditCEM(
        primitive, mode="decoder", candidates=4, iterations=1,
        elites=1, reachability_topk=1, low_horizon=2,
    )
    mask = vocab.token_to_id["<mask>"]
    result = planner.first_action(
        [[vocab.token_to_id["prompt"]]], [[mask, mask]]
    )
    assert result.first_action == result.decoded_actions[0]
    assert len(result.decoded_actions) == 2
