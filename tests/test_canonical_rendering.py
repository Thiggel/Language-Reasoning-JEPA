"""Canonical step rendering: temporary variable names must be deterministic.

The 'rand' symbol naming gave encoder targets an LN-L1 spread of ~half a
solution step (measured 2026-08-24) — pure label noise on the predictor.
FaithfulEnv now forces the reference renderer's 'seq' naming unless
TEXTJEPA_LEGACY_RENDER=1 reproduces the historical behavior.
"""

import random

import numpy as np

from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab


def _traj(fp, salt):
    random.seed(salt)
    np.random.seed(salt)
    env = fp.make_env()
    out, guard = [], 0
    while not env.solved and guard < 64:
        feas = env.feasible_actions()
        if not feas:
            break
        q = sorted([a for a in feas if a in fp.necessary] or list(feas))[0]
        out.append(env.step(q))
        guard += 1
    return out


def test_step_rendering_is_rng_independent(monkeypatch):
    monkeypatch.delenv("TEXTJEPA_LEGACY_RENDER", raising=False)
    ds = FaithfulDataset(cached_faithful_vocab(), size=2, seed=11,
                         max_op=15, max_edge=20, op_range=(3, 8),
                         distractor_prob=0.0)
    for i in range(2):
        fp, _ = ds.problem(i)
        assert _traj(fp, 1) == _traj(fp, 999)


def test_legacy_flag_restores_random_naming(monkeypatch):
    monkeypatch.setenv("TEXTJEPA_LEGACY_RENDER", "1")
    ds = FaithfulDataset(cached_faithful_vocab(), size=4, seed=11,
                         max_op=15, max_edge=20, op_range=(4, 10),
                         distractor_prob=0.0)
    diff = any(_traj(ds.problem(i)[0], 1) != _traj(ds.problem(i)[0], 999)
               for i in range(4))
    assert diff, "legacy mode should draw temp names from the global RNG"
