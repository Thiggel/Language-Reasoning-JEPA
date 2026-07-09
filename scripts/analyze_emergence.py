"""Answer-emergence curve: linear & MLP decodability of the FINAL answer
from intermediate states s_t, grouped by remaining necessary steps.

Usage: python scripts/analyze_emergence.py ckpt=runs/X/best.pt device=cuda:0
Writes <run>/emergence.json: {remaining: {"linear": acc, "mlp": acc, "n": n}}
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from textjepa.probing.extract import extract_features
from textjepa.probing.probes import logistic_probe_accuracy, mlp_probe_accuracy
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for, load_run


@hydra.main(config_path="../configs", config_name="probe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    ds = build_dataset(run_cfg, vocab, split="val", size=cfg.n_samples)
    loader = DataLoader(
        ds, batch_size=128, num_workers=4,
        collate_fn=partial(collate_for(run_cfg), pad_id=vocab.pad_id),
    )
    feats = extract_features(model, loader, cfg.device)
    x, y, rem = feats["state"], feats["answer_step"], feats["remaining"]
    out = {}
    for r in sorted(np.unique(rem)):
        sel = rem == r
        if sel.sum() < 400:
            continue
        xs, ys = x[sel], y[sel]
        if len(xs) > 8000:
            idx = np.random.RandomState(0).permutation(len(xs))[:8000]
            xs, ys = xs[idx], ys[idx]
        out[int(r)] = {
            "linear": round(logistic_probe_accuracy(xs, ys), 4),
            "mlp": round(mlp_probe_accuracy(xs, ys), 4),
            "n": int(sel.sum()),
        }
        print(r, out[int(r)])
    dest = Path(cfg.out or Path(cfg.ckpt).parent / "emergence.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
