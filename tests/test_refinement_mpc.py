import inspect

from scripts.plan_token_refinement_mpc import search_first_action
from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset, MASK_TOKEN, faithful_token_edit_vocab,
)
from textjepa.models.edit_jepa import EditJEPA


def test_refinement_beam_is_target_free_and_returns_replacement():
    assert "target" not in inspect.signature(search_first_action).parameters
    vocab = faithful_token_edit_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=1, seed=59, max_op=4, max_edge=8,
        op_range=(3, 4), corruption_mode="iterative_refinement",
        refinement_probability=0.25,
    )
    item = dataset[0]
    model = EditJEPA(
        len(vocab), vocab.pad_id, d_model=32, chunk_layers=1,
        chunk_heads=4, slot_layers=1, slot_heads=4, n_slots=2,
        predictor_layers=1, predictor_heads=4, max_chunk_len=320,
        max_buffer_len=16, d_action=8, macro_k=0, token_aligned=True,
        token_predictor_layers=1, dropout=0.0, refinement_prior=True,
    ).eval()
    action, scores, selected_gar = search_first_action(
        model, item["prompt"], item["buffers"][0], vocab.pad_id, "cpu",
        horizon=2, beam_width=2, top_positions=2, top_tokens=2,
        max_candidates=4, prior_weight=0.05, gar_weight=1.0,
        excluded_tokens={
            vocab.pad_id, vocab.token_to_id[vocab.UNK],
            vocab.token_to_id[MASK_TOKEN],
        },
    )
    assert action[0] == "replace"
    assert action in scores
    assert isinstance(selected_gar, float)
