from pathlib import Path


def test_oracle_distance_runner_pairs_three_levels_of_oracle_information():
    script = Path("scripts/run_intent_oracle_distance_controls.sh").read_text()
    assert "oracle_goal symbolic_oracle_goal symbolic_exact_distance" in script
    assert "1:64 2:8 2:64 4:8 4:64" in script
    assert 'SCALING_SLACKS="0"' in script
