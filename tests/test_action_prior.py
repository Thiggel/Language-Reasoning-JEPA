"""Learned Gaussian action prior: head, training signal, planner, validation.

The prior p(a | s) is a state-conditioned (mixture of) isotropic Gaussian(s)
over the intent-phrase embedding space, trained with the NLL of the observed
next action. At plan time (candidate_interface=learned_catalogue) it ranks
and filters the problem's full action catalogue — no feasible-action menu and
no feasibility oracle are consulted.
"""

import pytest
import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.models import DiscourseJEPA
from textjepa.models.heads import GaussianActionPrior
from textjepa.objectives import GaussianActionPriorNLL
from textjepa.planning import LatentPlanner
from textjepa.planning.search import validate_learned_catalogue_checkpoint


def _tiny_model(vocab, **kwargs) -> DiscourseJEPA:
    return DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=64,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, d_macro=4,
        **kwargs,
    )


# --------------------------------------------------------------------- #
# head
# --------------------------------------------------------------------- #
def test_gaussian_prior_shapes_for_single_and_mixture_components():
    for n_components in (1, 3):
        prior = GaussianActionPrior(16, 8, n_components=n_components)
        state = torch.randn(2, 5, 16)
        action = torch.randn(2, 5, 8)
        logits, mu, logvar = prior.components(state)
        assert logits.shape == (2, 5, n_components)
        assert mu.shape == (2, 5, n_components, 8)
        assert logvar.shape == (2, 5, n_components)
        assert prior.log_prob(state, action).shape == (2, 5)
        assert torch.allclose(
            prior.nll(state, action), -prior.log_prob(state, action)
        )


def test_gaussian_prior_rejects_zero_components():
    with pytest.raises(ValueError):
        GaussianActionPrior(16, 8, n_components=0)


def test_gaussian_prior_nll_decreases_on_toy_state_action_mapping():
    torch.manual_seed(0)
    states = torch.randn(64, 16)
    targets = torch.tanh(states @ torch.randn(16, 8))
    prior = GaussianActionPrior(16, 8, hidden=64)
    opt = torch.optim.Adam(prior.parameters(), lr=1e-2)
    initial = prior.nll(states, targets).mean().item()
    for _ in range(150):
        opt.zero_grad()
        loss = prior.nll(states, targets).mean()
        loss.backward()
        opt.step()
    assert prior.nll(states, targets).mean().item() < initial - 1.0


# --------------------------------------------------------------------- #
# model + objective
# --------------------------------------------------------------------- #
def test_model_emits_prior_nll_only_when_enabled_and_objective_backprops():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=2, seed=3)
    batch = collate([dataset[0], dataset[1]], vocab.pad_id)

    plain = _tiny_model(vocab)
    assert plain.action_prior is None
    out = plain(batch)
    assert "action_prior_nll" not in out.extras
    # objective is a no-op without the head (weight can stay in the config)
    assert float(GaussianActionPriorNLL()(out, batch)) == 0.0

    model = _tiny_model(
        vocab, action_prior=True, action_prior_states="all",
        action_prior_candidate_scope="catalogue",
    )
    out = model(batch)
    T = batch["step_mask"].shape[1]
    assert out.extras["action_prior_nll"].shape == (2, 3, T)
    assert out.extras["action_prior_nll_valid"].shape == (2, 3, T)
    loss = GaussianActionPriorNLL()(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for p in model.action_prior.parameters() if p.grad is not None
    ]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_model_rejects_unknown_prior_modes():
    vocab = build_vocab(23)
    with pytest.raises(ValueError):
        _tiny_model(vocab, action_prior_states="sometimes")
    with pytest.raises(ValueError):
        _tiny_model(vocab, action_prior_candidate_scope="everything")


# --------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------- #
def test_learned_catalogue_planner_runs_without_feasibility_oracle():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=1, seed=3)
    problem, _ = dataset.problem(0)
    model = _tiny_model(
        vocab, action_prior=True, action_prior_states="all",
        action_prior_candidate_scope="catalogue",
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=1,
        candidate_interface="learned_catalogue", prior_top_k=3,
    )
    with torch.no_grad():
        result = planner.plan_episode(problem, slack=1, seed=0)
    assert result.steps >= 1
    assert result.prior_rank is not None
    assert 1.0 <= result.prior_rank <= len(problem.vars)


def test_learned_catalogue_depth_two_expands_from_imagined_states():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=1, seed=5)
    problem, _ = dataset.problem(0)
    model = _tiny_model(
        vocab, action_prior=True, action_prior_states="all",
        action_prior_candidate_scope="catalogue",
    ).eval()
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"), lookahead=2, max_expand=4,
        candidate_interface="learned_catalogue", prior_top_k=2,
        search_algorithm="beam",
    )
    with torch.no_grad():
        result = planner.plan_episode(problem, slack=1, seed=0)
    assert result.steps >= 1


def test_prior_filters_are_validated_and_top_k_limits_candidates():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=1, seed=3)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab, action_prior=True).eval()
    with pytest.raises(ValueError):
        LatentPlanner(
            model, vocab, torch.device("cpu"),
            candidate_interface="learned_catalogue", prior_top_p=0.0,
        )
    with pytest.raises(ValueError):
        LatentPlanner(
            model, vocab, torch.device("cpu"),
            candidate_interface="learned_catalogue", prior_top_k=-1,
        )
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"),
        candidate_interface="learned_catalogue", prior_top_k=2,
    )
    with torch.no_grad():
        state = planner._s0(
            planner._tokens(["one plus one"]),
            torch.ones(1, 1, dtype=torch.bool),
        )
        candidates = planner._prior_candidates(problem, state)
    assert len(candidates) == 2
    nucleus = LatentPlanner(
        model, vocab, torch.device("cpu"),
        candidate_interface="learned_catalogue", prior_top_p=1e-9,
    )
    with torch.no_grad():
        assert len(nucleus._prior_candidates(problem, state)) == 1


def test_learned_catalogue_requires_prior_head_with_clear_error():
    vocab = build_vocab(23)
    dataset = IGSMDataset(vocab, size=1, seed=3)
    problem, _ = dataset.problem(0)
    model = _tiny_model(vocab).eval()  # trained without the prior head
    planner = LatentPlanner(
        model, vocab, torch.device("cpu"),
        candidate_interface="learned_catalogue",
    )
    with pytest.raises(RuntimeError, match="model.action_prior=true"):
        with torch.no_grad():
            planner.plan_episode(problem, slack=0, seed=0)


# --------------------------------------------------------------------- #
# checkpoint validation
# --------------------------------------------------------------------- #
def _valid_learned_catalogue_cfg():
    return OmegaConf.create({
        "model": {
            "action_prior": True,
            "action_support_states": "all",
            "action_prior_states": "all",
            "action_prior_candidate_scope": "catalogue",
        },
        "data": {"all_action_supervision": True},
        "objective": {
            "action_feasibility": {"weight": 1.0},
            "action_prior": {"weight": 1.0},
        },
    })


def test_learned_catalogue_checkpoint_validation_accepts_full_recipe():
    validate_learned_catalogue_checkpoint(_valid_learned_catalogue_cfg())


@pytest.mark.parametrize("path, value, message", [
    (("model", "action_prior"), False, "model.action_prior"),
    (("model", "action_prior_states"), "true", "action_prior_states"),
    (
        ("model", "action_prior_candidate_scope"),
        "feasible",
        "candidate_scope",
    ),
    (("objective", "action_prior", "weight"), 0.0, "action_prior.weight"),
])
def test_learned_catalogue_checkpoint_validation_rejects_missing_pieces(
    path, value, message
):
    cfg = _valid_learned_catalogue_cfg()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_learned_catalogue_checkpoint(cfg)
