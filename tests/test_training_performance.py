import torch

from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.predictor import CausalHistoryPredictor
from textjepa.training.loading import performance_loader_kwargs
from textjepa.training.optim import build_optimizer


def test_right_padding_does_not_change_valid_lm_logits():
    torch.manual_seed(3)
    model = DecoderLM(
        vocab_size=17, pad_id=0, d_model=16, n_layers=2, n_heads=4,
        ff_mult=2, max_len=8,
    ).eval()
    short = torch.tensor([[2, 4, 6]])
    padded = torch.tensor([[2, 4, 6, 0, 0]])
    with torch.no_grad():
        expected = model(short)
        actual = model(padded)[:, :3]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_right_padding_does_not_change_valid_predictor_outputs():
    torch.manual_seed(5)
    model = CausalHistoryPredictor(
        d_state=16, d_action=4, n_layers=2, n_heads=4,
        ff_mult=2, max_steps=8,
    ).eval()
    states = torch.randn(1, 3, 16)
    actions = torch.randn(1, 3, 4)
    padded_states = torch.cat([states, torch.randn(1, 2, 16)], 1)
    padded_actions = torch.cat([actions, torch.randn(1, 2, 4)], 1)
    with torch.no_grad():
        expected = model(states, actions, torch.ones(1, 3, dtype=torch.bool))
        actual = model(
            padded_states, padded_actions,
            torch.tensor([[True, True, True, False, False]]),
        )[:, :3]
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_loader_performance_kwargs_are_valid_with_and_without_workers():
    cpu = torch.device("cpu")
    assert performance_loader_kwargs(0, cpu) == {
        "pin_memory": False, "persistent_workers": False,
    }
    assert performance_loader_kwargs(3, cpu) == {
        "pin_memory": False, "persistent_workers": True, "prefetch_factor": 4,
    }


def test_cpu_optimizer_preserves_decay_partition_without_fused_mode():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, 0.1}
    assert not optimizer.defaults.get("fused", False)
