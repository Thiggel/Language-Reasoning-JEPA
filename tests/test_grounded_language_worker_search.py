from types import SimpleNamespace

import pytest
import torch

import textjepa.planning.grounded_language_worker as worker

from textjepa.models.hierarchical_language_jepa import (
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.planning.grounded_language_worker import (
    _rollout_token_endpoints,
    _root_token_history,
    build_optimized_worker_bank,
)


class NewlineTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [9]


class PrefixProposalFrozen:
    """Tiny causal proposal: first {1,2}, then newline or distractor."""

    def __call__(self, *, input_ids, output_hidden_states=False, **kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 19), -30.0)
        for row in range(batch):
            generated = input_ids[row, 3:]
            if len(generated) == 0:
                logits[row, -1, 1] = 10.0
                logits[row, -1, 2] = 9.0
            else:
                logits[row, -1, 9] = 10.0
                logits[row, -1, 8] = 9.0
        hidden = input_ids.float()[..., None].repeat(1, 1, 8) / 10
        return SimpleNamespace(logits=logits, hidden_states=[hidden])


def tiny_model():
    torch.manual_seed(7)
    return HierarchicalLanguageJEPA(HierarchicalLanguageJEPAConfig(
        d_backbone=8, vocab_size=19, pad_id=0,
        d_token=6, d_sentence=4, d_action=2, d_task=3,
        predictor_width=8, token_layers=1, sentence_layers=1,
        n_heads=2, token_context=4, sentence_context=4,
        max_span=4, enable_macro_actions=True,
    )).eval()


def test_beam_worker_optimizes_token_jepa_endpoint_not_qwen_likelihood():
    model = tiny_model()
    frozen = PrefixProposalFrozen()
    prefix = torch.tensor([3, 4, 5])
    hidden = frozen(
        input_ids=prefix[None], output_hidden_states=True
    ).hidden_states[-1][0]
    state_history, action_history = _root_token_history(
        model, hidden, prefix, prompt_len=3
    )
    desired = _rollout_token_endpoints(
        model, state_history, action_history, [torch.tensor([2, 9])]
    )[0]
    bank = build_optimized_worker_bank(
        model, frozen, NewlineTokenizer(), prefix, hidden, desired,
        lambda left, right: (left - right).square().sum(-1),
        prompt_len=3, algorithm="beam", objective="jepa",
        population=8, k0=2, iterations=2, elite_fraction=0.25,
        beam_width=2, branch_factor=2, preserve_prefix=1,
        prior_weight=0.0, temperature=0.8, top_p=0.95, top_k=0,
        seed=1,
    )
    assert bank.candidates[0][0].tolist() == [2, 9]
    assert bank.search_algorithm == "beam"
    assert bank.proposed_tokens == 6
    torch.testing.assert_close(bank.predicted_coarse[0], desired)


@pytest.mark.parametrize("algorithm", ["markov_cem", "factorized_cem"])
def test_population_worker_reports_missing_complete_support_as_runtime_failure(
    monkeypatch, algorithm
):
    model = tiny_model()
    frozen = PrefixProposalFrozen()
    prefix = torch.tensor([3, 4, 5])
    hidden = frozen(
        input_ids=prefix[None], output_hidden_states=True
    ).hidden_states[-1][0]
    monkeypatch.setattr(
        worker, "generate_complete_reasoning_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("no complete candidates")
        ),
    )
    with pytest.raises(RuntimeError, match="no supported complete proposal"):
        build_optimized_worker_bank(
            model, frozen, NewlineTokenizer(), prefix, hidden,
            torch.zeros(model.config.d_sentence),
            lambda left, right: (left - right).square().sum(-1),
            prompt_len=3, algorithm=algorithm, objective="jepa",
            population=4, k0=2, iterations=1, elite_fraction=0.25,
            beam_width=2, branch_factor=2, preserve_prefix=1,
            prior_weight=0.0, temperature=0.8, top_p=0.95, top_k=0,
            seed=1,
        )
