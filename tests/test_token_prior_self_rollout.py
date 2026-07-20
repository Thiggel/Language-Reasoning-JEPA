from types import SimpleNamespace

import torch

from scripts.train_token_hierarchy_v2 import token_prior_self_rollout_loss
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def test_self_rollout_prior_loss_is_finite_and_updates_prior():
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=24, pad_id=0, d_model=16, encoder_layers=1,
        predictor_layers=1, n_heads=2, ff_mult=2, max_len=64,
        d_action=8, level_spans=[4], level_dims=[6],
        variational_levels=[False], concat_width=2, use_token_prior=True,
        token_prior_detach_state=True,
    )
    tokens = torch.randint(1, 24, (3, 20))
    out = model(tokens, torch.tensor([8, 8, 8]))
    obj = SimpleNamespace(
        token_prior_self_rollout=1.0,
        token_prior_self_rollout_depth=3,
        token_prior_self_rollout_policy="greedy",
        token_prior_self_rollout_topk=4,
        token_prior_self_rollout_temperature=1.0,
        token_prior_self_rollout_detach_state=True,
        token_prior_label_smoothing=0.0,
    )
    loss, metrics = token_prior_self_rollout_loss(model, out, obj)
    loss.backward()
    assert torch.isfinite(loss)
    assert "token_prior_self_rollout_h3" in metrics
    assert any(parameter.grad is not None for parameter in model.token_prior.parameters())
