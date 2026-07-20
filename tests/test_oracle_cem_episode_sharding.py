from pathlib import Path


def test_oracle_cem_supports_disjoint_episode_shards():
    source = Path("scripts/plan_token_hierarchy_oracle_cem.py").read_text()
    assert '"--episode-offset"' in source
    assert "args.episode_offset + args.episodes" in source
    assert '"episode_indices"' in source


def test_bottleneck_runner_forwards_episode_offset():
    source = Path("scripts/run_token_bottleneck_ladder.sh").read_text()
    assert "TOKEN_LADDER_EPISODE_OFFSET" in source
    assert '--episode-offset "$episode_offset"' in source
