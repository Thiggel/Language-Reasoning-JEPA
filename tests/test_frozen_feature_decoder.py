import torch

from textjepa.analysis.reconstruction import FrozenFeatureDecoder


def test_frozen_decoder_never_backpropagates_into_source_features():
    features = torch.randn(4, 6, requires_grad=True)
    targets = torch.tensor([[1, 2, 0], [2, 1, 0], [1, 1, 0], [2, 2, 0]])
    model = FrozenFeatureDecoder(6, 4, pad_id=0, hidden_dim=8)
    loss = model.loss(features, targets)
    loss.backward()
    assert features.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())
