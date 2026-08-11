"""Shared plumbing for the frozen state -> sentence read-out.

Both ``scripts/train_state_decoder.py`` and ``scripts/eval_state_decoder.py``
need the same three things: a genuinely frozen backbone, a cheap iGSM loader
(the geometric-ranking supervision fields are training-only and very slow to
generate, so they are switched off here), and the true/imagined state tensors.
Nothing in this module ever touches a backbone gradient.
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import collate
from textjepa.utils.checkpoint import build_dataset, load_run


def load_frozen(ckpt: str, device: str):
    """``load_run`` + hard freeze; asserts no backbone parameter trains."""
    model, vocab, cfg = load_run(ckpt, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters()), (
        "backbone parameters must not require grad"
    )
    return model, vocab, cfg


def readout_cfg(cfg):
    """Copy of ``cfg`` with training-only dataset supervision disabled."""
    lean = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(lean):
        lean.data.geo_rank_k = 0
        lean.data.geo_rank_factual_only = False
        lean.data.geo_rank_horizons = None
        lean.data.geo_rank_rollouts = 1
        lean.data.dense_geo_anchors = False
        lean.data.n_alt = 0
        lean.data.macro_alt_k = 0
        lean.data.all_action_supervision = False
    return lean


def make_loader(cfg, vocab, split: str, size: int, batch_size: int,
                shuffle: bool = False, workers: int = 2) -> DataLoader:
    dataset = build_dataset(readout_cfg(cfg), vocab, split=split, size=size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=lambda b: collate(b, vocab.pad_id),
    )


@torch.no_grad()
def true_states(model, batch: dict) -> torch.Tensor:
    """[B, T, D] encoder states; ``states[:, t]`` has consumed step ``t``."""
    _, states = model.encode_states(
        batch["prompt_tokens"], batch["prompt_mask"],
        batch["step_tokens"], batch["step_mask"],
    )
    return states


@torch.no_grad()
def imagined_states(model, batch: dict, depth: int) -> torch.Tensor:
    """[B, depth, D] predictor rollout from ``s_0`` along the true actions.

    ``s_0`` is the prompt-only state; applying the ground-truth action codes
    one at a time gives the model's *imagined* state after ``d`` steps, which
    the read-out then renders.  Same rollout call as the planner
    (``model.predictor(state, action_code)``).
    """
    s0, _ = model.encode_states(
        batch["prompt_tokens"], batch["prompt_mask"],
        batch["step_tokens"], batch["step_mask"],
    )
    codes = model.encode_actions(batch["action_tokens"])
    current = s0
    out = []
    for d in range(min(depth, codes.shape[1])):
        current = model.predictor(current, codes[:, d])
        out.append(current)
    return torch.stack(out, dim=1)
