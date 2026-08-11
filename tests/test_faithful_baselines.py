"""Faithful competitor baselines: TD-JEPA successor features and GoalHead."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.models import DiscourseJEPA
from textjepa.models.heads import (
    GoalHead,
    StateFeatureHead,
    SuccessorFeatureHead,
    TaskEmbeddingHead,
    ridge_reward_projection,
)
from textjepa.objectives import GoalHeadDistill, TDJEPASuccessor
from textjepa.planning import LatentPlanner


def _tiny_model(**overrides) -> DiscourseJEPA:
    kwargs = dict(
        vocab_size=len(build_vocab(23)), pad_id=build_vocab(23).pad_id,
        d_model=64, chunk_layers=1, chunk_heads=2,
        state_layers=2, state_heads=2, d_action=8, d_macro=4,
        predictor_kind="concat", macro_k=0,
    )
    kwargs.update(overrides)
    return DiscourseJEPA(**kwargs)


def _tiny_batch(vocab, geo_rank_k: int = 0):
    dataset = IGSMDataset(
        vocab, size=2, seed=11, geo_rank_k=geo_rank_k,
        geo_rank_horizon=1,
    )
    return collate([dataset[0], dataset[1]], vocab.pad_id)


# --------------------------------------------------------------------- #
# head shapes
# --------------------------------------------------------------------- #
def test_successor_feature_head_shapes_and_task_broadcasting():
    head = SuccessorFeatureHead(16, 4, 6, d_psi=8)
    state = torch.randn(3, 5, 16)
    action = torch.randn(3, 5, 4)
    task = torch.randn(3, 6)
    assert head(state, action, task).shape == (3, 5, 8)
    assert head(state[:, 0], action[:, 0], task).shape == (3, 8)


def test_state_feature_and_task_embedding_head_shapes():
    psi = StateFeatureHead(16, d_psi=8)
    assert psi(torch.randn(3, 5, 16)).shape == (3, 5, 8)
    tau = TaskEmbeddingHead(16, d_task=6)
    assert tau(torch.randn(3, 16)).shape == (3, 6)


def test_goal_head_shape_preserves_state_width():
    head = GoalHead(16)
    assert head(torch.randn(3, 16)).shape == (3, 16)


# --------------------------------------------------------------------- #
# td_jepa objective
# --------------------------------------------------------------------- #
def _fake_out(extras):
    return SimpleNamespace(
        extras=extras, step_states=torch.zeros(1, 1, 2, requires_grad=True)
    )


def test_td_jepa_objective_terminal_handling_and_masking():
    gamma = 0.9
    pred = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [5.0, 5.0]]])
    next_features = torch.tensor([[[1.0, 1.0], [2.0, 0.0], [9.0, 9.0]]])
    next_pred = torch.tensor([[[2.0, 0.0], [4.0, 4.0], [9.0, 9.0]]])
    valid = torch.tensor([[True, True, False]])
    terminal = torch.tensor([[False, True, False]])
    out = _fake_out(dict(
        td_jepa_pred=pred, td_jepa_next_features=next_features,
        td_jepa_next_pred=next_pred, td_valid=valid, td_terminal=terminal,
    ))
    loss = TDJEPASuccessor(gamma=gamma)(out, {})
    # step 0 bootstraps: target = psi' + 0.9 * T' = [2.8, 1.0];
    # error mean over d_psi = ((2.8-1)^2 + 1^2) / 2.
    # terminal step 1: target = psi' only = [2, 0]; error = (2^2 + 0) / 2.
    # padded step (all 5s/9s) must not contribute.
    step0 = ((2.8 - 1.0) ** 2 + 1.0 ** 2) / 2
    step1 = (2.0 ** 2 + 0.0) / 2
    torch.testing.assert_close(loss, torch.tensor((step0 + step1) / 2))
    assert torch.isfinite(loss)


def test_td_jepa_objective_is_noop_without_supervision():
    assert TDJEPASuccessor()(_fake_out({}), {}).item() == 0.0


# --------------------------------------------------------------------- #
# z_r ridge regression
# --------------------------------------------------------------------- #
def test_ridge_reward_projection_recovers_known_z():
    torch.manual_seed(0)
    features = torch.randn(200, 8)
    z_true = torch.randn(8)
    rewards = features @ z_true
    z_hat = ridge_reward_projection(features, rewards, eps=1e-4)
    torch.testing.assert_close(z_hat, z_true, atol=1e-3, rtol=1e-3)


def test_model_fit_td_jepa_reward_projection_sets_buffer():
    vocab = build_vocab(23)
    model = _tiny_model(geo_rank_score_mode="td_jepa")
    assert not bool(model.core.td_jepa_z_r_fitted)
    model.fit_td_jepa_reward_projection([_tiny_batch(vocab)])
    assert bool(model.core.td_jepa_z_r_fitted)
    assert torch.isfinite(model.core.td_jepa_z_r).all()
    assert model.core.td_jepa_z_r.abs().sum() > 0


# --------------------------------------------------------------------- #
# goal_head objective
# --------------------------------------------------------------------- #
def test_goal_head_loss_is_zero_when_prediction_equals_target():
    target = torch.randn(4, 16)
    exact = _fake_out(dict(goal_head_pred=target, goal_head_target=target))
    torch.testing.assert_close(
        GoalHeadDistill()(exact, {}), torch.tensor(0.0)
    )
    off = _fake_out(dict(
        goal_head_pred=-target, goal_head_target=target
    ))
    assert GoalHeadDistill()(off, {}).item() > 0.0


def test_goal_head_loss_weights_cosine_term():
    pred = torch.tensor([[0.0, 2.0]])
    target = torch.tensor([[2.0, 0.0]])  # orthogonal: 1 - cos = 1
    base = GoalHeadDistill(cos_weight=0.0)(_fake_out(dict(
        goal_head_pred=pred, goal_head_target=target
    )), {})
    weighted = GoalHeadDistill(cos_weight=2.0)(_fake_out(dict(
        goal_head_pred=pred, goal_head_target=target
    )), {})
    torch.testing.assert_close(weighted - base, torch.tensor(2.0))


# --------------------------------------------------------------------- #
# model supervision emission and gating
# --------------------------------------------------------------------- #
def test_model_emits_td_jepa_supervision_with_terminal_masks():
    vocab = build_vocab(23)
    model = _tiny_model(geo_rank_score_mode="td_jepa")
    batch = _tiny_batch(vocab)
    out = model(batch)
    B, T = batch["step_mask"].shape
    d_psi = model.core.td_jepa_z_r.shape[0]
    assert out.extras["td_jepa_pred"].shape == (B, T, d_psi)
    assert out.extras["td_jepa_next_features"].shape == (B, T, d_psi)
    assert out.extras["td_jepa_next_pred"].shape == (B, T, d_psi)
    assert torch.equal(out.extras["td_valid"], batch["step_mask"])
    last = batch["step_mask"].sum(1) - 1
    for row in range(B):
        expected = torch.zeros(T, dtype=torch.bool)
        expected[last[row]] = True
        assert torch.equal(out.extras["td_terminal"][row], expected)
    loss = TDJEPASuccessor()(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [
        p.grad for p in model.core.successor_feature_head.parameters()
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    task_grads = [
        p.grad for p in model.core.td_jepa_task_head.parameters()
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in task_grads)
    # psi appears only under no_grad (sg in the loss): it must stay fixed.
    assert all(
        p.grad is None for p in model.core.state_feature_head.parameters()
    )


def test_model_emits_goal_head_supervision_toward_ema_terminal_state():
    vocab = build_vocab(23)
    model = _tiny_model(geo_rank_score_mode="goal_head")
    batch = _tiny_batch(vocab)
    out = model(batch)
    B, T = batch["step_mask"].shape
    assert out.extras["goal_head_pred"].shape == (B, 64)
    last = batch["step_mask"].sum(1) - 1
    expected = out.step_states_tgt[torch.arange(B), last]
    torch.testing.assert_close(out.extras["goal_head_target"], expected)
    loss = GoalHeadDistill()(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.core.goal_head.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("mode", ["td_jepa", "goal_head"])
def test_score_mode_gates_only_matching_heads(mode):
    model = _tiny_model(geo_rank_score_mode=mode)
    td_jepa = all(
        p.requires_grad
        for head in (
            model.core.successor_feature_head,
            model.core.td_jepa_task_head,
            model.core.state_feature_head,
        )
        for p in head.parameters()
    )
    goal = all(p.requires_grad for p in model.core.goal_head.parameters())
    assert td_jepa == (mode == "td_jepa")
    assert goal == (mode == "goal_head")
    assert not any(p.requires_grad for p in model.core.value_head.parameters())
    assert not any(p.requires_grad for p in model.core.td_q_head.parameters())


@pytest.mark.parametrize("mode", ["td_jepa", "goal_head"])
def test_forward_with_geo_rank_data_does_not_crash(mode):
    vocab = build_vocab(23)
    model = _tiny_model(geo_rank_score_mode=mode)
    batch = _tiny_batch(vocab, geo_rank_k=2)
    out = model(batch)
    if mode == "goal_head" and "ga_energy" in out.extras:
        assert torch.isfinite(out.extras["ga_energy"]).all()
    if mode == "td_jepa":
        # No training-time scalar energy exists before z_r is fitted.
        assert "ga_energy" not in out.extras


# --------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------- #
class _AddPredictor(nn.Module):
    def forward(self, state, action):
        return state + action


def test_planner_td_jepa_cost_is_negative_q_of_pre_final_state_and_action():
    class T(nn.Module):
        def forward(self, state, action, task):
            return (10 * state[..., :1] + action[..., :1])

    class Tau(nn.Module):
        def forward(self, initial):
            return initial

    model = SimpleNamespace(
        predictor=_AddPredictor(), geo_rank_score_mode="td_jepa",
        core=SimpleNamespace(
            macro_k=0, d_action=1,
            successor_feature_head=T(), td_jepa_task_head=Tau(),
            td_jepa_z_r=torch.tensor([1.0]),
            td_jepa_z_r_fitted=torch.tensor(True),
        ),
    )
    planner = LatentPlanner(
        model, None, torch.device("cpu"), lookahead=2,
        allow_oracle_future_actions=True,
    )
    planner._action_codes = lambda _problem, actions: torch.tensor(
        actions, dtype=torch.float32
    ).unsqueeze(-1)
    costs = planner._flat_costs(
        torch.zeros(1, 1), torch.zeros(1, 1), None,
        [[1, 2], [3, None]], None,
    )
    # [1, 2]: pre-final state 1, final action 2 -> T = 12, Q = T^T z_r = 12.
    # [3, None]: absorbing pad, length 1: pre-final is the root (0),
    # final action 3 -> Q = 3.  Costs are -Q.
    torch.testing.assert_close(costs, torch.tensor([-12.0, -3.0]))


def test_planner_td_jepa_requires_fitted_reward_projection():
    model = SimpleNamespace(
        predictor=_AddPredictor(), geo_rank_score_mode="td_jepa",
        core=SimpleNamespace(
            macro_k=0, d_action=1,
            successor_feature_head=nn.Identity(),
            td_jepa_task_head=nn.Identity(),
            td_jepa_z_r=torch.zeros(1),
            td_jepa_z_r_fitted=torch.tensor(False),
        ),
    )
    planner = LatentPlanner(model, None, torch.device("cpu"))
    planner._action_codes = lambda _problem, actions: torch.tensor(
        actions, dtype=torch.float32
    ).unsqueeze(-1)
    with pytest.raises(RuntimeError, match="fit_td_jepa_reward_projection"):
        planner._flat_costs(
            torch.zeros(1, 1), torch.zeros(1, 1), None, [[1]], None
        )


def test_planner_goal_head_cost_is_ln_l1_distance_to_predicted_goal():
    goal = torch.tensor([[3.0, 1.0]])

    class G(nn.Module):
        def forward(self, initial):
            return goal.expand(initial.shape[0], -1)

    model = SimpleNamespace(
        predictor=_AddPredictor(), geo_rank_score_mode="goal_head",
        core=SimpleNamespace(macro_k=0, d_action=2, goal_head=G()),
    )
    planner = LatentPlanner(model, None, torch.device("cpu"), lookahead=1)
    planner._action_codes = lambda _problem, actions: torch.tensor(
        [[float(a), 0.0] for a in actions]
    )
    costs = planner._flat_costs(
        torch.zeros(1, 2), torch.zeros(1, 2), None, [[1], [2]], None
    )
    ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
    endpoints = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    expected = (ln(endpoints) - ln(goal)).abs().mean(-1)
    torch.testing.assert_close(costs, expected)


@pytest.mark.parametrize("mode", ["td_jepa", "goal_head"])
def test_planner_runs_end_to_end_with_faithful_modes(mode):
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=29).problem(0)
    model = _tiny_model(geo_rank_score_mode=mode).eval()
    if mode == "td_jepa":
        model.fit_td_jepa_reward_projection([_tiny_batch(vocab)])
    for lookahead in (1, 2):
        planner = LatentPlanner(
            model, vocab, torch.device("cpu"), lookahead=lookahead,
            max_expand=8, allow_oracle_future_actions=lookahead > 1,
        )
        result = planner.plan_episode(problem, slack=1, seed=5)
        assert isinstance(result.solved, bool)
        assert result.steps <= problem.n_necessary_steps + 1


def test_faithful_shuffle_actions_permutes_alignment_only():
    from textjepa.data.faithful import FaithfulDataset

    vocab = build_vocab(23)
    aligned = FaithfulDataset(vocab, size=6, seed=1)
    shuffled = FaithfulDataset(vocab, size=6, seed=1, shuffle_actions=True)
    changed = False
    for i in range(6):
        a, b = aligned[i], shuffled[i]
        # Every non-action field is untouched by the control.
        assert a["steps"] == b["steps"]
        assert a["prompt"] == b["prompt"]
        # Same multiset of action phrases, possibly permuted.
        assert sorted(map(tuple, a["actions"])) == sorted(
            map(tuple, b["actions"])
        )
        if a["actions"] != b["actions"]:
            changed = True
    assert changed, "shuffle_actions was a no-op on the faithful dataset"
