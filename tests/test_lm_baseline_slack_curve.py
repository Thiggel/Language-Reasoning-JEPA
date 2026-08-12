"""LM-baseline planning evaluations must emit the JEPA slack-curve metrics.

The paper contract reports success at slack {0,1,2,4}, area under that curve,
invalid-action rate, and mean excess actions among solved episodes for EVERY
row.  The LM baselines used to emit only a single fixed-slack success, so they
were not comparable with scripts/plan.py output.  These tests pin the shared
metric shape and the greedy-policy ``solved_at`` convention the curve needs.
"""

from __future__ import annotations

import ast
from pathlib import Path

from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.search import EpisodeResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
JEPA_CURVE_FIELDS = {"success_by_slack", "excess_steps"}


def _episode(solved: bool, steps: int, n_necessary: int) -> EpisodeResult:
    # Greedy LM policies stop the instant the goal is reached, so the executed
    # step count is also the solved-at step.  This is the convention both
    # plan_lm.py and plan_sentlm.py must use for the curve to be exact.
    return EpisodeResult(
        solved, steps, n_necessary, 0, 0,
        solved_at=steps if solved else None,
    )


def test_greedy_solved_at_convention_gives_exact_slack_curve():
    results = [
        _episode(True, 3, 3),    # excess 0 -> counts at every slack
        _episode(True, 5, 3),    # excess 2 -> counts from slack 2
        _episode(False, 7, 3),   # never solved -> counts nowhere
        _episode(True, 4, 3),    # excess 1 -> counts from slack 1
    ]
    metrics = aggregate_episodes(results, slack_curve=True, slack=4)
    assert metrics["success_by_slack"] == {
        "0": 0.25, "1": 0.5, "2": 0.75, "3": 0.75, "4": 0.75,
    }
    assert metrics["excess_steps"] == [0, 2, None, 1]
    # Success at the generous budget must agree with the curve's last point.
    assert metrics["success"] == metrics["success_by_slack"]["4"]


def test_aggregate_supplies_every_contract_field():
    metrics = aggregate_episodes(
        [_episode(True, 3, 3)], slack_curve=True, slack=4
    )
    for field in (
        "success", "mean_steps", "mean_necessary", "distractor_rate",
        "invalid_action_rate", *JEPA_CURVE_FIELDS,
    ):
        assert field in metrics, field


def _calls_aggregate(script: str) -> ast.Module:
    tree = ast.parse((SCRIPTS / script).read_text())
    names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "aggregate_episodes" in names, (
        f"{script} must aggregate through the shared evaluator so its JSON "
        "carries the same fields as scripts/plan.py"
    )
    return tree


def test_lm_planners_use_the_shared_evaluator_and_solved_at():
    for script in ("plan_lm.py", "plan_sentlm.py"):
        tree = _calls_aggregate(script)
        source = ast.unparse(tree)
        # A slack curve is scored from one generous-budget run, so it must not
        # overwrite the fixed-slack JSON at the same nominal slack.
        assert "_slackcurve" in source, script
        assert "solved_at" in source, script
        # Historical field name must survive as an alias for old collectors.
        assert "invalid_rate" in source, script
