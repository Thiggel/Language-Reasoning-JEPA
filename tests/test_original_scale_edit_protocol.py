from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    MASK_TOKEN,
    faithful_replacement_vocab,
)
from textjepa.models.multiscale_edit_jepa import (
    SentencePatternActionEncoder,
    TokenReplacementPrior,
)
from textjepa.utils.checkpoint import build_vocab_for_config


def test_replacement_vocab_has_no_rendered_actions_or_large_coordinates():
    vocab = faithful_replacement_vocab()
    assert MASK_TOKEN in vocab.token_to_id
    assert "position" not in vocab.token_to_id
    assert "with" not in vocab.token_to_id
    assert "417" not in vocab.token_to_id


def test_one_item_is_one_independent_replacement_transition():
    vocab = faithful_replacement_vocab()
    dataset = FaithfulTokenEditDataset(
        vocab, size=2, seed=211, max_op=6, max_edge=12,
        op_range=(3, 6), corruption_mode="iterative_refinement",
        sample_transition=True, content_only_actions=True,
    )
    item = dataset[0]
    assert len(item["buffers"]) == 2
    assert len(item["actions"]) == len(item["op"]) == 1
    assert item["actions"][0] == item["edit_content_token"]
    before = [x for sentence in item["buffers"][0] for x in sentence]
    after = [x for sentence in item["buffers"][1] for x in sentence]
    position = item["edit_position"][0]
    assert before[:position] == after[:position]
    assert before[position + 1:] == after[position + 1:]
    assert after[position] == item["edit_content_token"][0]


def test_sentence_pattern_encodes_relative_slot_but_not_absolute_coordinate():
    torch.manual_seed(3)
    encoder = SentencePatternActionEncoder(16, 4, n_heads=4)
    states = torch.randn(2, 7, 16)
    mask = torch.ones(2, 7, dtype=torch.bool)
    # Same five-token sentence and same relative slot, shifted by two tokens
    # in the flattened buffer.  Absolute flattened location must not matter.
    sentence_ids = torch.tensor([
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 1],
    ])
    positions = torch.tensor([2, 4])
    content = torch.randn(1, 16).expand(2, -1)
    code = encoder(
        states, mask, sentence_ids, torch.full((2,), 2), positions, content
    )
    assert torch.allclose(code[0], code[1], atol=1e-6)
    moved = encoder(
        states[:1], mask[:1], sentence_ids[:1], torch.tensor([2]),
        torch.tensor([3]), content[:1],
    )
    assert not torch.allclose(code[:1], moved)


def test_content_prior_scores_every_slot_without_position_head():
    prior = TokenReplacementPrior(16, 31, detach_state=True,
                                  predict_position=False)
    position, content = prior(
        torch.randn(3, 5, 16), torch.ones(3, 5, dtype=torch.bool),
        torch.randn(3, 16), torch.tensor([0, 2, 4]),
    )
    assert position is None
    assert content.shape == (3, 5, 31)


def test_three_jepas_match_the_mdlm_parameter_budget_within_one_percent():
    root = str(Path(__file__).parents[1] / "configs")
    target = 123_792_144  # 12-layer, width-912, context-768 MDLM control
    with initialize_config_dir(config_dir=root, version_base="1.3"):
        for name in (
            "edit_igsm_original_token",
            "edit_igsm_original_sentence",
            "edit_igsm_original_token_sentence",
        ):
            cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
            vocab = build_vocab_for_config(cfg)
            model = instantiate(cfg.model, vocab_size=len(vocab),
                                pad_id=vocab.pad_id)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert abs(count - target) / target < 0.01
