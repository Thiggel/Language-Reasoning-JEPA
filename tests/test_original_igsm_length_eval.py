import argparse
import importlib.util
from pathlib import Path

import pytest

from textjepa.data.faithful_token_edits import faithful_replacement_vocab
from textjepa.models.multiscale_edit_jepa import MultiscaleEditJEPA


SCRIPT = Path(__file__).parents[1] / "scripts" / "eval_original_igsm_lengths.py"
SPEC = importlib.util.spec_from_file_location("eval_original_igsm_lengths", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL)


def _small_model(vocab):
    return MultiscaleEditJEPA(
        len(vocab), vocab.pad_id, "token", d_model=16, d_action=4,
        token_layers=1, sentence_layers=1, predictor_layers=1, n_heads=4,
        max_sequence_len=32, max_sentences=4, base_prior=True,
        base_prior_predict_position=False,
    ).eval()


def test_mdlm_checkpoint_reconstruction_preserves_fractional_ff_multiplier():
    saved = argparse.Namespace(
        d_model=896, layers=12, heads=14, ff_mult=4.22,
        max_sequence_len=1024, attention_backend="flash",
        sequence_packing=True,
    )
    vocab = faithful_replacement_vocab()
    kwargs = EVAL.mdlm_model_kwargs(
        saved,
        {"vocab_size": len(vocab), "pad_id": vocab.pad_id, "mask_id": 3},
        vocab,
    )
    assert kwargs["ff_mult"] == 4.22
    assert kwargs["sequence_packing"] is True


def test_jepa_planner_modes_share_proposals_but_change_scoring():
    vocab = faithful_replacement_vocab()
    model = _small_model(vocab)
    prior = EVAL.make_jepa_planner(
        model, vocab, "cpu", "prior", candidate_budget=4, beam_width=2
    )
    gar = EVAL.make_jepa_planner(
        model, vocab, "cpu", "gar", candidate_budget=4, beam_width=2
    )
    assert prior.max_candidates == gar.max_candidates == 4
    assert prior.top_positions == gar.top_positions == 4
    assert prior.top_tokens == gar.top_tokens == 4
    assert prior.action_value_weight == prior.state_value_weight == 0.0
    assert gar.action_value_weight == 1.0
    assert gar.state_value_weight == 0.25


def test_prior_mode_refuses_a_q_based_refinement_stop():
    vocab = faithful_replacement_vocab()
    with pytest.raises(ValueError, match="requires GAR"):
        EVAL.evaluate_jepa(
            _small_model(vocab), vocab, [], "cpu", scoring="prior",
            stop_on_nonpositive_q=True,
        )


def test_oracle_candidate_diagnostic_does_not_mutate_current_buffer():
    vocab = faithful_replacement_vocab()
    model = _small_model(vocab)
    planner = EVAL.make_jepa_planner(
        model, vocab, "cpu", "gar", candidate_budget=4, beam_width=2
    )
    prompt = [[vocab.token_to_id["0"]]]
    mask = vocab.token_to_id["<mask>"]
    current = [[mask, mask]]
    target = [[vocab.token_to_id["1"], vocab.token_to_id["2"]]]
    before = [list(sentence) for sentence in current]
    diagnostic = EVAL.initial_candidate_diagnostics(
        planner, prompt, current, target
    )
    assert current == before
    assert diagnostic["proposal_count"] <= 4
    assert diagnostic["proposal_correct_action_recall"] in {0.0, 1.0}
    assert diagnostic["oracle_candidate_available"] in {0.0, 1.0}
