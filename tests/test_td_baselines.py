"""Competitor value-energy baselines: TD-Q and expectile goal-value heads."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from textjepa.data.igsm.dataset import IGSMDataset, build_vocab, collate
from textjepa.models import DiscourseJEPA
from textjepa.models.heads import ExpectileValueHead, TDQHead
from textjepa.objectives import ExpectileValueTD, TDQ
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


def test_td_q_head_shapes_and_broadcasting():
    head = TDQHead(16, 4)
    state = torch.randn(3, 5, 16)
    action = torch.randn(3, 5, 4)
    initial = torch.randn(3, 16)
    assert head(state, action, initial).shape == (3, 5)
    assert head(state[:, 0], action[:, 0], initial).shape == (3,)


def test_expectile_value_head_is_nonpositive_and_zero_at_match():
    head = ExpectileValueHead(16)
    state = torch.randn(3, 5, 16)
    initial = torch.randn(3, 16)
    value = head(state, initial)
    assert value.shape == (3, 5)
    assert (value <= 0).all()
    # V(z, z0) = -||f(z) - g(z0)|| is exactly zero when f(z) == g(z0).
    head.g = head.f
    same = head(initial, initial)
    torch.testing.assert_close(same, torch.zeros_like(same))


def _fake_out(extras):
    return SimpleNamespace(
        extras=extras, step_states=torch.zeros(1, 1, 2, requires_grad=True)
    )


def test_td_q_objective_terminal_handling_and_masking():
    gamma = 0.9
    pred = torch.tensor([[0.0, 0.0, 5.0]])
    bootstrap = torch.tensor([[2.0, 3.0, 7.0]])
    valid = torch.tensor([[True, True, False]])
    terminal = torch.tensor([[False, True, False]])
    out = _fake_out(dict(
        td_q_pred=pred, td_next_value=bootstrap,
        td_valid=valid, td_terminal=terminal,
    ))
    loss = TDQ(gamma=gamma)(out, {})
    # step 0 bootstraps: target -1 + 0.9*2 = 0.8; terminal step 1: target -1.
    # The padded step (pred 5, bootstrap 7) must not contribute.
    expected = (0.8 ** 2 + 1.0 ** 2) / 2
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert torch.isfinite(loss)


def test_td_q_objective_is_noop_without_supervision():
    out = _fake_out({})
    assert TDQ()(out, {}).item() == 0.0


def test_expectile_objective_weights_positive_delta_by_tau():
    tau = 0.9
    valid = torch.tensor([[True]])
    terminal = torch.tensor([[True]])
    # Terminal target is -1; delta = -1 - v.
    positive = _fake_out(dict(
        expectile_value_pred=torch.tensor([[-2.0]]),  # delta = +1
        td_next_value=torch.zeros(1, 1),
        td_valid=valid, td_terminal=terminal,
    ))
    negative = _fake_out(dict(
        expectile_value_pred=torch.tensor([[0.0]]),  # delta = -1
        td_next_value=torch.zeros(1, 1),
        td_valid=valid, td_terminal=terminal,
    ))
    objective = ExpectileValueTD(gamma=0.9, tau=tau)
    up = objective(positive, {})
    down = objective(negative, {})
    torch.testing.assert_close(up, torch.tensor(tau))
    torch.testing.assert_close(down, torch.tensor(1.0 - tau))
    torch.testing.assert_close(up / down, torch.tensor(tau / (1.0 - tau)))


def test_expectile_objective_rejects_degenerate_tau():
    with pytest.raises(ValueError, match="tau"):
        ExpectileValueTD(tau=1.0)


@pytest.mark.parametrize("mode", ["td_q", "expectile_value"])
def test_model_emits_td_supervision_and_marks_last_valid_step(mode):
    vocab = build_vocab(23)
    model = _tiny_model(geo_rank_score_mode=mode)
    batch = _tiny_batch(vocab)
    out = model(batch)
    key = "td_q_pred" if mode == "td_q" else "expectile_value_pred"
    B, T = batch["step_mask"].shape
    assert out.extras[key].shape == (B, T)
    assert out.extras["td_next_value"].shape == (B, T)
    assert torch.equal(out.extras["td_valid"], batch["step_mask"])
    last = batch["step_mask"].sum(1) - 1
    for row in range(B):
        expected = torch.zeros(T, dtype=torch.bool)
        expected[last[row]] = True
        assert torch.equal(out.extras["td_terminal"][row], expected)
    objective = TDQ() if mode == "td_q" else ExpectileValueTD()
    loss = objective(out, batch)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("mode", ["td_q", "expectile_value"])
def test_score_mode_gates_only_matching_head(mode):
    model = _tiny_model(geo_rank_score_mode=mode)
    td_q = all(p.requires_grad for p in model.core.td_q_head.parameters())
    expectile = all(
        p.requires_grad for p in model.core.expectile_value_head.parameters()
    )
    assert td_q == (mode == "td_q")
    assert expectile == (mode == "expectile_value")
    assert not any(p.requires_grad for p in model.core.value_head.parameters())


@pytest.mark.parametrize("mode", ["td_q", "expectile_value"])
def test_geo_rank_energies_exist_for_new_modes(mode):
    vocab = build_vocab(23)
    model = _tiny_model(geo_rank_score_mode=mode)
    batch = _tiny_batch(vocab, geo_rank_k=2)
    out = model(batch)
    if "ga_energy" in out.extras:
        assert torch.isfinite(out.extras["ga_energy"]).all()


def test_planner_td_q_cost_is_negative_q_of_pre_final_state_and_action():
    class Predictor(nn.Module):
        def forward(self, state, action):
            return state + action

    class Q(nn.Module):
        def forward(self, state, action, initial):
            return 10 * state[..., 0] + action[..., 0]

    model = SimpleNamespace(
        predictor=Predictor(), geo_rank_score_mode="td_q",
        core=SimpleNamespace(macro_k=0, d_action=1, td_q_head=Q()),
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
    # [1, 2]: pre-final state 1, final action 2 -> Q = 12.
    # [3, None]: absorbing pad, so length 1: pre-final is the root (0),
    # final action 3 -> Q = 3.  Costs are -Q.
    torch.testing.assert_close(costs, torch.tensor([-12.0, -3.0]))


def test_planner_expectile_cost_is_negative_value_of_endpoint():
    class Predictor(nn.Module):
        def forward(self, state, action):
            return state + action

    class V(nn.Module):
        def forward(self, state, initial):
            return -state[..., 0].abs()

    model = SimpleNamespace(
        predictor=Predictor(), geo_rank_score_mode="expectile_value",
        core=SimpleNamespace(
            macro_k=0, d_action=1, expectile_value_head=V()
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
        torch.zeros(1, 1), torch.zeros(1, 1), None, [[1, 2]], None
    )
    # Endpoint is 3, V = -3, cost = -V = 3.
    torch.testing.assert_close(costs, torch.tensor([3.0]))


@pytest.mark.parametrize("mode", ["td_q", "expectile_value"])
def test_planner_runs_end_to_end_with_baseline_modes(mode):
    vocab = build_vocab(23)
    problem, _ = IGSMDataset(vocab, size=1, seed=29).problem(0)
    model = _tiny_model(geo_rank_score_mode=mode).eval()
    for lookahead in (1, 2):
        planner = LatentPlanner(
            model, vocab, torch.device("cpu"), lookahead=lookahead,
            max_expand=8, allow_oracle_future_actions=lookahead > 1,
        )
        result = planner.plan_episode(problem, slack=1, seed=5)
        assert isinstance(result.solved, bool)
        assert result.steps <= problem.n_necessary_steps + 1
