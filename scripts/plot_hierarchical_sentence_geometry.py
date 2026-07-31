#!/usr/bin/env python3
"""Plot controlled sentence-endpoint geometry using t-SNE and optional UMAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _project(values: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(30.0, max(2.0, (len(values) - 1) / 3))
        return TSNE(
            n_components=2, init="pca", learning_rate="auto",
            perplexity=perplexity, random_state=seed,
        ).fit_transform(values)
    if method == "umap":
        try:
            import umap
        except ImportError as error:
            raise RuntimeError(
                "UMAP requested but umap-learn is not installed; t-SNE remains available"
            ) from error
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(values)
    raise ValueError(f"unknown projection method {method}")


def _plot(coordinates, group_ids, variant_ids, sources, title, output):
    import matplotlib.pyplot as plt
    colors = {0: "#2468a2", 1: "#31a354", 2: "#de2d26"}
    markers = {0: "o", 1: "^", 2: "x"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for axis, source in zip(axes, ("igsm_controlled", "logical_ood")):
        keep = np.asarray(sources) == source
        for group in np.unique(group_ids[keep]):
            index = np.where((group_ids == group) & keep)[0]
            anchor = index[variant_ids[index] == 0][0]
            for variant, style, color in ((1, "-", "#31a354"), (2, "--", "#de2d26")):
                other = index[variant_ids[index] == variant][0]
                axis.plot(
                    coordinates[[anchor, other], 0], coordinates[[anchor, other], 1],
                    style, color=color, alpha=0.25, linewidth=0.8,
                )
        for variant, label in enumerate(("anchor", "paraphrase", "contrast")):
            mask = keep & (variant_ids == variant)
            axis.scatter(
                coordinates[mask, 0], coordinates[mask, 1],
                s=20 if source == "logical_ood" else 9,
                c=colors[variant], marker=markers[variant], alpha=0.72,
                linewidths=0.8, label=label,
            )
        axis.set(title=source.replace("_", " "), xticks=[], yticks=[])
    axes[1].legend(loc="best", fontsize=8)
    figure.suptitle(title)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("tsne", "umap"), action="append")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    methods = args.method or ["tsne"]
    payload = torch.load(args.artifact, map_location="cpu", weights_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = payload["group_ids"].numpy()
    variants = payload["variant_ids"].numpy()
    sources = [row["source"] for row in payload["metadata"]]
    manifest = {"artifact": str(args.artifact), "projections": []}
    for space, tensor in payload["representations"].items():
        values = tensor.float().numpy()
        for method in methods:
            coordinates = _project(values, method, args.seed)
            stem = f"sentence_geometry_{space}_{method}"
            np.savez_compressed(
                args.output_dir / f"{stem}.npz",
                coordinates=coordinates.astype(np.float32),
                group_ids=groups, variant_ids=variants,
                sources=np.asarray(sources), method=method, space=space,
            )
            _plot(
                coordinates, groups, variants, sources,
                f"{space}: {method.upper()} (qualitative)",
                args.output_dir / f"{stem}.png",
            )
            manifest["projections"].append({"space": space, "method": method})
    (args.output_dir / "sentence_geometry_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
