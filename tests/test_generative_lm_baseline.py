import torch

from scripts.eval_generative_lm_baseline import sentence_next_logits
from scripts.train_lm import lm_loss
from textjepa.data.igsm.dataset import build_vocab
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM


def test_requested_lm_baselines_are_ten_million_parameters():
    vocab = build_vocab(23)
    token = DecoderLM(len(vocab), vocab.pad_id, 320, 8, 8, 4, 768)
    sentence = SentenceLM(
        len(vocab), vocab.pad_id, 304, 2, 4, 4, 8, 2, 4, 4, 96, 64, False
    )
    for model in (token, sentence):
        count = sum(parameter.numel() for parameter in model.parameters())
        assert 9_500_000 <= count <= 10_500_000


def test_sentence_baseline_has_only_decoder_ce_loss():
    vocab = build_vocab(23)
    model = SentenceLM(
        len(vocab), vocab.pad_id, 32, 1, 4, 1, 4, 1, 4, 2, 16, 8, False
    )
    batch = {
        "prompt_tokens": torch.tensor([[[1, 2, 3]]]),
        "prompt_mask": torch.tensor([[True]]),
        "step_tokens": torch.tensor([[[4, 5, 6], [7, 8, 9]]]),
        "step_mask": torch.tensor([[True, True]]),
    }
    assert set(model(batch)) == {"ce"}


def test_sentence_next_logits_depend_only_on_supplied_prefix():
    vocab = build_vocab(23)
    model = SentenceLM(
        len(vocab), vocab.pad_id, 32, 1, 4, 1, 4, 1, 4, 2, 16, 8, False
    ).eval()
    context = torch.randn(1, 32)
    first = sentence_next_logits(model, context, [[1, 2]])
    batched = sentence_next_logits(model, context, [[1, 2], [1, 3]])[0:1]
    torch.testing.assert_close(first, batched)


def test_token_lm_masks_prompt_cross_entropy():
    vocab = build_vocab(23)
    model = DecoderLM(len(vocab), vocab.pad_id, 32, 1, 4, 2, 16)
    batch = {
        "tokens": torch.tensor([[1, 2, 3, 4, 5]]),
        "prompt_len": torch.tensor([3]),
    }
    loss = lm_loss(model, batch, torch.device("cpu"))
    assert torch.isfinite(loss)
