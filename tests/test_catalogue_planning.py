from pathlib import Path

from textjepa.data.planbench import compile_blocksworld_episode, parse_blocksworld_pddl
from textjepa.data.proofwriter import compile_proofwriter_episode
from textjepa.data.observed_action import build_observed_action_vocab
from textjepa.models import DiscourseJEPA
from textjepa.planning.catalogue import (
    CatalogueLatentPlanner,
    FullCatalogueLatentPlanner,
    environment_from_episode,
    environment_from_faithful_problem,
)
import torch
from omegaconf import OmegaConf

from tests.test_planbench_adapter import PDDL
from tests.test_proofwriter_adapter import _record
from scripts.eval_observed_action import OracleReplayPolicy


def test_blocksworld_environment_executes_catalogue_actions_and_counts_invalid():
    episode = compile_blocksworld_episode(
        parse_blocksworld_pddl(PDDL), "test"
    )
    environment = environment_from_episode(episode)
    assert not environment.solved
    before = environment.invalid_actions
    environment.step("stack block a on block b")
    assert environment.invalid_actions == before + 1
    for transition in episode.transitions:
        environment.step(transition.action)
    assert environment.solved


def test_oracle_reference_replays_expert_without_oracle_menu():
    episode = compile_blocksworld_episode(
        parse_blocksworld_pddl(PDDL), "test"
    )
    result = OracleReplayPolicy().run_episode(
        environment_from_episode(episode), excess_actions=0
    )
    assert result.solved
    assert result.steps == result.optimal_length
    assert result.invalid_actions == 0


def test_alfworld_pilot_repeats_fixed_episodes_across_epochs():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "configs/data/alfworld_pilot.yaml")
    assert cfg.name == "observed_action"
    assert cfg.fresh_per_epoch is False


def test_paper_geometry_gar_configuration_has_no_learned_candidate_prior():
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        root / "configs/experiment/paper_causal_geometry_gar_no_prior.yaml"
    )
    assert cfg.objective.action_feasibility.weight == 0.0
    assert cfg.model.action_support_states == "none"
    assert cfg.data.all_action_supervision is False
    assert cfg.data.dense_geo_anchors is True


def test_proofwriter_environment_supports_alternative_valid_derivations():
    episode = compile_proofwriter_episode(_record(), "Q1", "test")
    environment = environment_from_episode(episode)
    for transition in episode.transitions:
        environment.step(transition.action)
    assert environment.solved
    assert environment.invalid_actions == 0


def test_catalogue_planner_runs_depth_two_without_querying_oracle_menu():
    episode = compile_blocksworld_episode(
        parse_blocksworld_pddl(PDDL), "test"
    )
    vocab = build_observed_action_vocab([episode])
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        max_chunk_len=64, max_chunks=32,
    ).eval()
    planner = CatalogueLatentPlanner(
        model, vocab, torch.device("cpu"), simulation_depth=2,
        proposal_top_m=2, beam_width=2,
    )
    result = planner.run_episode(environment_from_episode(episode), 0)
    assert result.steps <= result.optimal_length
    assert result.invalid_actions >= 0


def test_full_catalogue_planner_reranks_every_action_without_support_head():
    class ForbiddenSupport:
        def __call__(self, *args, **kwargs):
            raise AssertionError("full-catalogue JEPA consulted support head")

    class Value:
        def __call__(self, leaf, goal):
            return leaf[:, 0]

    class Model:
        value_head = Value()
        core = type("Core", (), {"action_support_head": ForbiddenSupport()})()

    planner = FullCatalogueLatentPlanner(
        Model(), None, torch.device("cpu"), simulation_depth=1,
        beam_width=1,
    )
    planner._observed_history = lambda *args: (
        torch.zeros(1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 0, 1)
    )
    planner._action_codes = lambda catalogue: torch.tensor([[2.0], [0.0]])
    planner._rollout = lambda state, action, future: future[:, -1]
    assert planner.choose((), [], [], ("worse", "better")) == "better"


def test_faithful_igsm_wrapper_exposes_full_catalogue_not_feasible_menu():
    from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab

    dataset = FaithfulDataset(
        cached_faithful_vocab(), size=1, seed=917, max_op=15,
        max_edge=20, op_range=(3, 5), distractor_prob=0.0,
    )
    problem, _ = dataset.problem(0)
    environment = environment_from_faithful_problem(problem)
    assert len(environment.catalogue) == len(problem.action_order)
    assert len(environment.environment.feasible_actions()) < len(
        environment.catalogue
    )
