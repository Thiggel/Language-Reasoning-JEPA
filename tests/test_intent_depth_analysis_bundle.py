from pathlib import Path


def test_bundle_prioritizes_causal_depth_audits_before_seed_completion():
    script = Path("scripts/run_intent_depth_analysis_bundle.sh").read_text()
    assert script.index("run_intent_planning_depth_audit.sh") < script.index(
        "run_intent_decision_audit_cell.sh"
    )
    assert '"full:$full_checkpoint" "direct:$direct_checkpoint"' in script
    assert 'EPISODES="$depth_episodes"' in script
    assert 'EPISODES="$geometry_episodes"' in script
