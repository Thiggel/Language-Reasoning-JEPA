"""Menu-free (full_catalogue) LM baselines on FAITHFUL iGSM.

The paper's main results are menu-free on the faithful domain, so the token-LM
and sentence-LM planning evaluators must support candidate_interface=
full_catalogue there: candidates are the problem's whole action catalogue
(``faithful_catalogue``), invalid picks execute through ``step_or_invalid``
and are counted in ``invalid_action_rate``, and the policy's own attempted
actions are masked with the same state-scoped semantics as FaithfulPlanner
(reset on progress) -- without it a deterministic argmax loops on a no-op.

These tests run the real scripts end-to-end on tiny untrained checkpoints.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from textjepa.data.faithful import cached_faithful_vocab
from textjepa.data.igsm.dataset import build_vocab
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM

REPO = Path(__file__).resolve().parents[1]

FAITHFUL_DATA = {
    "name": "igsm_real",
    "max_op": 15, "max_edge": 20, "op_range": [3, 10],
    "distractor_prob": 0.0, "max_distractors": 0,
    "train_size": 4, "train_seed": 0, "val_size": 4, "val_seed": 3,
    "geo_rank_candidate_interface": "feasible_menu",
    # keys the shared dataset-builder reads before branching to faithful
    "modulus": 23, "n_vars_range": [4, 6], "leaf_prob": 0.5,
    "steps_range": [3, 6],
}

STYLIZED_DATA = {
    "name": "igsm",
    "modulus": 7, "n_vars_range": [4, 6], "leaf_prob": 0.5,
    "steps_range": [3, 5], "distractor_prob": 0.0, "max_distractors": 0,
    "train_size": 4, "train_seed": 0, "val_size": 4, "val_seed": 3,
}

TOK_MODEL = {"d_model": 32, "n_layers": 1, "n_heads": 2, "max_len": 2048}
SENT_MODEL = {
    "d_model": 32, "chunk_layers": 1, "chunk_heads": 2, "state_layers": 1,
    "state_heads": 2, "dec_layers": 1, "dec_heads": 2, "max_chunk_len": 128,
    "max_chunks": 96,
}


def _write_ckpt(tmp_path: Path, kind: str, data: dict) -> Path:
    torch.manual_seed(0)
    faithful = data["name"] == "igsm_real"
    vocab = cached_faithful_vocab() if faithful else build_vocab(
        data["modulus"]
    )
    if kind == "tok":
        model_cfg = TOK_MODEL
        model = DecoderLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **model_cfg
        )
    else:
        model_cfg = SENT_MODEL
        model = SentenceLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **model_cfg
        )
    cfg = {
        "data": data, "model": model_cfg, "train": {"target_kind": "intent"},
    }
    path = tmp_path / f"{kind}_{data['name']}.pt"
    torch.save(
        {"model": model.state_dict(),
         "cfg": OmegaConf.to_container(OmegaConf.create(cfg))},
        path,
    )
    return path


def _run(script: str, ckpt: Path, out: Path, *overrides: str) -> dict:
    cmd = [
        sys.executable, str(REPO / "scripts" / script),
        f"ckpt={ckpt}", "device=cpu", "n_episodes=2", "slack=2",
        f"out={out}", *overrides,
    ]
    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True, timeout=1200
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text())


CONTRACT_FIELDS = (
    "success", "mean_steps", "mean_necessary", "distractor_rate",
    "invalid_action_rate", "invalid_rate", "candidate_interface",
)


@pytest.mark.parametrize("script,key", [
    ("plan_lm.py", "lm_intent_policy"),
    ("plan_sentlm.py", "sentlm_intent_decoder"),
])
def test_faithful_full_catalogue_runs_end_to_end(tmp_path, script, key):
    kind = "tok" if script == "plan_lm.py" else "sent"
    ckpt = _write_ckpt(tmp_path, kind, FAITHFUL_DATA)
    metrics = _run(
        script, ckpt, tmp_path / "masked.json",
        "candidate_interface=full_catalogue", "slack_frac=0.5",
        "eval_necessary_range=[2,8]",
    )[key]
    for field in CONTRACT_FIELDS:
        assert field in metrics, field
    assert metrics["candidate_interface"] == "full_catalogue"
    assert metrics["invalid_rate"] == metrics["invalid_action_rate"]
    # An untrained policy over the whole catalogue must hit invalid actions,
    # and the noop executor must count them.
    assert metrics["invalid_action_rate"] > 0.0

    # The state-scoped attempted mask must be live: the unmasked ablation
    # lets the deterministic argmax lock onto the same infeasible action, so
    # the two runs cannot coincide and masking can only reduce invalids.
    unmasked = _run(
        script, ckpt, tmp_path / "unmasked.json",
        "candidate_interface=full_catalogue", "slack_frac=0.5",
        "eval_necessary_range=[2,8]", "mask_attempted=false",
    )[key]
    assert unmasked != metrics
    assert metrics["invalid_action_rate"] <= unmasked["invalid_action_rate"]


@pytest.mark.parametrize("script,key", [
    ("plan_lm.py", "lm_intent_policy"),
    ("plan_sentlm.py", "sentlm_intent_decoder"),
])
def test_stylized_paths_unchanged(tmp_path, script, key):
    kind = "tok" if script == "plan_lm.py" else "sent"
    ckpt = _write_ckpt(tmp_path, kind, STYLIZED_DATA)
    menu = _run(script, ckpt, tmp_path / "menu.json")[key]
    assert menu["invalid_action_rate"] == 0.0
    # slack_frac defaults to 0 and mask_attempted has no effect on the
    # stylized catalogue: explicit defaults must reproduce the default run.
    full = _run(
        script, ckpt, tmp_path / "full.json",
        "candidate_interface=full_catalogue",
    )[key]
    full_explicit = _run(
        script, ckpt, tmp_path / "full2.json",
        "candidate_interface=full_catalogue", "slack_frac=0.0",
        "mask_attempted=true",
    )[key]
    assert full == full_explicit
    assert full["candidate_interface"] == "full_catalogue"
