"""The code_prior planning interface must be strictly opt-in."""

import pytest
import torch

from textjepa.planning.flat_search import (
    CANDIDATE_INTERFACES, MENU_FREE_INTERFACES, PROPOSER_INTERFACES,
    FlatPlanner,
)


class _Stub:
    observed_action_decoder = None


def _planner(**kw):
    return FlatPlanner(_Stub(), None, torch.device("cpu"), **kw)


def test_code_prior_is_registered_as_a_menu_free_proposer():
    assert "code_prior" in CANDIDATE_INTERFACES
    assert "code_prior" in MENU_FREE_INTERFACES
    assert "code_prior" in PROPOSER_INTERFACES


def test_code_prior_refuses_to_run_without_both_artifacts():
    with pytest.raises(RuntimeError, match="code_prior needs BOTH"):
        _planner(candidate_interface="code_prior")
    with pytest.raises(RuntimeError, match="code_prior needs BOTH"):
        _planner(candidate_interface="code_prior", code_prior=object())
    with pytest.raises(RuntimeError, match="code_prior needs BOTH"):
        _planner(candidate_interface="code_prior", action_decoder=object())


def test_defaults_are_off_for_every_existing_interface():
    for iface in CANDIDATE_INTERFACES:
        if iface in {"code_prior", "flow_rerank", "flow_decode", "ldad_cycle"}:
            continue
        p = _planner(candidate_interface=iface)
        assert p.code_prior is None and p.action_decoder is None


def test_generate_outcomes_default_reproduces_historical_behaviour():
    """None must mean: generated outcomes iff the interface is autonomous."""
    assert _planner(candidate_interface="autonomous").generate_outcomes is True
    for iface in ("feasible_menu", "full_catalogue", "prior_propose"):
        assert _planner(candidate_interface=iface).generate_outcomes is False


def test_generate_outcomes_can_be_forced_either_way():
    assert _planner(candidate_interface="feasible_menu",
                    generate_outcomes=True).generate_outcomes is True
    assert _planner(candidate_interface="autonomous",
                    generate_outcomes=False).generate_outcomes is False
