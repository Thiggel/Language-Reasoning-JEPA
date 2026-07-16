"""Probe a trained checkpoint. Usage:

    python scripts/probe.py ckpt=runs/my_run/best.pt
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.probing.extract import extract_features
from textjepa.probing.suite import EDIT_PROBE_TASKS, run_probe_suite
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, collate_for, load_run


def probe_model(model, vocab, cfg, device, n_samples, batch_size=128, seed=0, mlp=False):
    # Preference/counterfactual candidates create extra model outputs but do
    # not affect the encoded demonstration states used by any probe.  Longer
    # GAR or bounded-beam teachers can otherwise dominate probe runtime.
    analysis_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    analysis_cfg.data.geo_rank_k = 0
    analysis_cfg.data.n_alt = 0
    ds = build_dataset(analysis_cfg, vocab, split="val", size=n_samples)
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=4,
        collate_fn=partial(collate_for(analysis_cfg), pad_id=vocab.pad_id),
    )
    feats = extract_features(model, loader, device)
    tasks = EDIT_PROBE_TASKS if cfg.data.get("name", "igsm") == "igsm_edit" else None
    return run_probe_suite(feats, seed=seed, tasks=tasks, mlp=mlp)


@hydra.main(config_path="../configs", config_name="probe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    df = probe_model(model, vocab, run_cfg, cfg.device, cfg.n_samples, seed=cfg.seed,
                     mlp=cfg.get("mlp", False))
    df = df.rename(columns={"acc": "acc_trained"})

    if cfg.random_control:
        rnd, _, _ = load_run(cfg.ckpt, cfg.device, random_init=True)
        df_r = probe_model(rnd, vocab, run_cfg, cfg.device, cfg.n_samples, seed=cfg.seed)
        df["acc_random_enc"] = df_r["acc"]

    cols = ["task", "acc_trained"] + (
        ["acc_mlp"] if "acc_mlp" in df.columns else []
    ) + (
        ["acc_random_enc"] if cfg.random_control else []
    ) + ["majority", "n", "description"]
    df = df[cols]
    out = Path(cfg.out or Path(cfg.ckpt).parent / "probe_results.csv")
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
