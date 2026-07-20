from functools import partial

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from scripts.eval_flat_sentence_planning import beam_plan
from scripts.train_flat_sentence_jepa import compute_losses
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.flat_sentence_jepa import FlatSentenceJEPA


def _batch(size=3):
    vocab = build_vocab(23)
    dataset = SemanticBoundaryLMDataset(
        vocab, size=size, seed=31, boundary_mode="semantic", modulus=23,
        n_vars_range=(8, 10), leaf_prob=0.35, steps_range=(4, 6),
        distractor_prob=0.0, max_distractors=0,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=size,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )))
    return batch, vocab


def _model(vocab, prior=True, detach=False):
    return FlatSentenceJEPA(
        len(vocab), vocab.pad_id, d_state=24, encoder_layers=1,
        predictor_layers=1, n_heads=4, ff_mult=2, max_len=768,
        d_action=8, dense_depth=3, use_token_prior=prior,
        token_prior_detach_state=detach,
    )


def test_flat_sentence_transition_is_one_token_shift_in_causal_sentence_space():
    batch, vocab = _batch(2)
    model = _model(vocab).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"])
    for row in range(2):
        prompt = int(batch["prompt_len"][row])
        count = int(out["lengths"][row])
        assert torch.allclose(out["prev"][row, 0], out["states"][row, prompt - 1])
        assert torch.allclose(
            out["target"][row, :count],
            out["targets"][row, prompt:prompt + count],
        )
    assert model.predictor.causal_sequence


def test_token_counterfactual_factual_candidate_matches_ema_next_state():
    batch, vocab = _batch(3)
    model = _model(vocab).eval()
    with torch.no_grad():
        out = model(batch["tokens"], batch["prompt_len"])
        cf = model.token_counterfactuals(
            out, batch["tokens"], batch["prompt_len"], k=4, max_anchors=7
        )
    assert cf["value"].shape == (7, 4)
    anchors = cf["anchor_indices"]
    factual = out["target"][anchors[:, 0], anchors[:, 1]]
    assert torch.allclose(cf["exact_outcome"][:, 0], factual, atol=1e-5)
    assert model.token_value.net[0].normalized_shape == (
        2 * model.d_state + model.d_action,
    )


def test_detached_prior_does_not_send_gradient_into_state_encoder():
    batch, vocab = _batch(2)
    model = _model(vocab, prior=True, detach=True).train()
    out = model(batch["tokens"], batch["prompt_len"])
    loss = torch.nn.functional.cross_entropy(
        out["token_prior_logits"][out["valid"]], out["action_ids"][out["valid"]]
    )
    loss.backward()
    assert not any(p.grad is not None and p.grad.abs().sum() for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.token_prior.parameters())


def test_beam_depth_changes_rollout_length_without_symbolic_filtering():
    _, vocab = _batch(1)
    model = _model(vocab).eval()
    prompt = [vocab.token_to_id.get("<bos>", 1), 2, 3]
    goal = torch.randn(1, model.d_state)
    result = beam_plan(
        model, prompt, goal, depth=4, width=8, score_mode="oracle",
        proposal_mode="all", proposal_topk=16,
    )
    assert len(result["tokens"]) == 4
    assert result["expanded"] > 0
    assert result["uses_symbolic_feasibility"] is False
    prior = beam_plan(
        model, prompt, goal, depth=1, width=1, score_mode="prior",
        proposal_mode="prior", proposal_topk=8,
    )
    assert len(prior["tokens"]) == 1


def test_joint_prediction_dense_gar_and_prior_objective_backpropagates():
    batch, vocab = _batch(3)
    model = _model(vocab).train()
    cfg = OmegaConf.load("configs/flat_sentence_jepa.yaml")
    out = model(batch["tokens"], batch["prompt_len"])
    loss, items = compute_losses(out, cfg, model, batch)
    assert torch.isfinite(loss)
    assert {
        "prediction", "dense", "token_prior", "gar_regression",
        "gar_ranking", "gar_counterfactual_mse",
    } <= items.keys()
    loss.backward()
    for module in (model.encoder, model.predictor, model.token_value):
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )
