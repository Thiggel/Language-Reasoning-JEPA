from pathlib import Path


def test_scaling_runner_keeps_checkpoint_fixed_and_labels_privilege():
    script = Path("scripts/run_intent_fixed_checkpoint_scaling.sh").read_text()
    assert 'sha256sum "$checkpoint"' in script
    assert 'SCALING_CELLS:-"1:64 2:64 4:64 8:64' in script
    assert "allow_oracle_future_actions=$oracle" in script
    assert "symbolic-future-action-tree" in script
    assert 'measure_flops=true' in script
    assert 'compute_out=${stem}_compute.json' in script
    assert "symbolic_oracle_goal" in script
    assert "symbolic_exact_distance" in script
    assert '"energy=$energy"' in script


def test_plan_compute_metadata_is_a_sidecar():
    plan = Path("scripts/plan.py").read_text()
    config = Path("configs/plan.yaml").read_text()
    assert "compute_out: null" in config
    assert '"checkpoint_sha256": digest' in plan
    assert '"oracle_future_action_tree"' in plan
    assert '"energy": str(cfg.get("energy", "value"))' in plan
    assert "measured_flops_per_episode" in plan
    assert "results[" not in plan
