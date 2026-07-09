"""Visualize learned latent geometry. Usage:

    python scripts/analyze.py ckpt=runs/disc_base/best.pt

Produces PNGs next to the checkpoint: PCA of states/deltas/actions colored
by ground-truth structure, value calibration, and rollout drift by horizon.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for, load_run


def pca2(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def scatter(ax, xy, labels, title, cmap="tab10"):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=4, cmap=cmap, alpha=0.6)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


@hydra.main(config_path="../configs", config_name="probe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    device = torch.device(cfg.device)
    ds = build_dataset(run_cfg, vocab, split="val", size=cfg.n_samples)
    loader = DataLoader(
        ds, batch_size=128, num_workers=4,
        collate_fn=partial(collate_for(run_cfg), pad_id=vocab.pad_id),
    )

    states, deltas, actions, rollout, tgt = [], [], [], [], []
    ops, remaining, necessary, values, vpred, vtrue, horizons = [], [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            m = out.step_mask
            fm = m.reshape(-1)
            flat = lambda x: x.reshape(-1, x.shape[-1])[fm].cpu().numpy()
            states.append(flat(out.step_states))
            deltas.append(flat(out.step_states - out.prev_states))
            actions.append(flat(out.actions))
            rollout.append(flat(out.rollout))
            tgt.append(flat(out.step_states_tgt))
            for k, acc in (("op", ops), ("remaining", remaining),
                           ("necessary", necessary), ("value", values)):
                acc.append(batch[k].reshape(-1)[fm].cpu().numpy())
            h = torch.arange(1, m.shape[1] + 1, device=device).expand_as(m)
            horizons.append(h.reshape(-1)[fm].cpu().numpy())
            vm = torch.cat([torch.ones_like(m[:, :1]), m], 1).reshape(-1)
            vpred.append(out.value_pred.reshape(-1)[vm].cpu().numpy())
            rem = torch.cat([batch["n_necessary"].unsqueeze(1), batch["remaining"]], 1)
            vtrue.append(rem.reshape(-1)[vm].cpu().numpy())

    cat = lambda xs: np.concatenate(xs)
    states, deltas, actions, rollout, tgt = map(cat, (states, deltas, actions, rollout, tgt))
    ops, remaining, necessary, values = map(cat, (ops, remaining, necessary, values))
    vpred, vtrue, horizons = map(cat, (vpred, vtrue, horizons))
    n = min(len(states), 4000)
    idx = np.random.RandomState(0).permutation(len(states))[:n]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    scatter(axes[0, 0], pca2(states[idx]), remaining[idx], "states, colored by remaining steps")
    scatter(axes[0, 1], pca2(deltas[idx]), ops[idx], "state displacements Δs, colored by op")
    scatter(axes[0, 2], pca2(actions[idx]), ops[idx], "action codes, colored by op")
    scatter(axes[1, 0], pca2(deltas[idx]), necessary[idx], "Δs, colored by goal-relevance", cmap="coolwarm")
    scatter(axes[1, 1], pca2(states[idx]), values[idx] % 10, "states, colored by step value (mod 10)")

    ax = axes[1, 2]
    jitter = np.random.RandomState(0).normal(0, 0.08, len(vtrue))
    ax.scatter(vtrue + jitter, vpred, s=3, alpha=0.25)
    lim = max(vtrue.max(), 1)
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("true remaining")
    ax.set_ylabel("predicted remaining")
    ax.set_title("value head calibration", fontsize=9)

    out_dir = Path(cfg.ckpt).parent
    fig.tight_layout()
    fig.savefig(out_dir / "geometry.png", dpi=140)

    # rollout drift: normalized L1 to EMA target by horizon
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    def ln(x):
        mu = x.mean(-1, keepdims=True)
        sd = x.std(-1, keepdims=True) + 1e-6
        return (x - mu) / sd
    drift = np.abs(ln(rollout) - ln(tgt)).mean(-1)
    hs = sorted(set(horizons.tolist()))
    ax2.plot(hs, [drift[horizons == h].mean() for h in hs], marker="o")
    ax2.set_xlabel("rollout horizon (steps)")
    ax2.set_ylabel("L1 to EMA target (normalized)")
    ax2.set_title("open-loop rollout drift", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_dir / "rollout_drift.png", dpi=140)
    print(f"saved {out_dir}/geometry.png and rollout_drift.png")


if __name__ == "__main__":
    main()
