"""Boundary tests for the predeclared experiment-selection rules."""

from scripts.buildup_decision import decision as buildup_decision
from scripts.report_ablation_matrix import fmt_loss
from scripts.recipe_report import matched_reference_name
from scripts.selector_decision import decision as selector_decision
from scripts.shear_decision import decision as shear_decision


def test_shear_decision_boundaries():
    assert shear_decision(0.02, 0.02, n_seeds=1) == "remove"
    assert shear_decision(0.03, 0.00, n_seeds=1) == "second seed"
    assert shear_decision(0.051, 0.00, n_seeds=1) == "retain"
    assert shear_decision(0.03, 0.00, n_seeds=2) == "retain"


def test_buildup_decision_boundaries():
    assert buildup_decision(0.02, 0.02, n_seeds=1) == "reject"
    assert buildup_decision(0.03, -0.02, n_seeds=1) == "second seed"
    assert buildup_decision(0.051, 0.00, n_seeds=1) == "add"
    assert buildup_decision(0.03, 0.00, n_seeds=2) == "add"
    assert buildup_decision(0.10, -0.021, n_seeds=1) == "reject"


def test_selector_decision_boundaries():
    assert selector_decision(0.02, 0.02, 1) == "retain H=2/B=1"
    assert selector_decision(0.03, -0.02, 1) == "second seed"
    assert selector_decision(0.051, 0.00, 1) == "advance"
    assert selector_decision(0.03, 0.00, 2) == "advance"
    assert selector_decision(0.10, -0.021, 1) == "reject"


def test_recipe_report_matches_reference_seed_suffix():
    reference = "disc_latent_goal_h2_r1"
    assert matched_reference_name(reference, "ablation") == reference
    assert matched_reference_name(reference, "ablation_s2") == f"{reference}_s2"
    assert matched_reference_name(reference, "ablation_s12") == f"{reference}_s12"


def test_ablation_report_uses_paired_mean_loss():
    assert fmt_loss([0.8, 0.7], [0.75, 0.72]) == "+0.015"
    assert fmt_loss([0.8], []) == "---"
