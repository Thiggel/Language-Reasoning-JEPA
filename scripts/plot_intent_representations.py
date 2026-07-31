"""Create regeneratable PCA/t-SNE/UMAP views of an exported intent space.

The coordinate archive is the scientific artifact; figures are only a view of
it.  This lets paper styling, point subsampling, and label choices change
without touching a checkpoint or recomputing frozen representations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _sample(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= limit:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=limit, replace=False)


def _save_plot(coords: np.ndarray, labels: np.ndarray, title: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    for label in np.unique(labels):
        mask = labels == label
        axis.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.55,
                     label=str(label), linewidths=0)
    axis.set(title=title, xticks=[], yticks=[])
    axis.legend(title="label", markerscale=2, fontsize=7, loc="best")
    figure.savefig(out, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="NPZ from export_intent_representations")
    parser.add_argument("--label", default="categorical_necessary")
    parser.add_argument("--method", choices=("pca", "tsne", "umap"), default="pca")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coordinates-out", required=True)
    parser.add_argument("--figure-out")
    args = parser.parse_args()

    source = np.load(args.features)
    if args.label not in source:
        raise KeyError(f"missing label {args.label}; available: {source.files}")
    indices = _sample(len(source["representations"]), args.samples, args.seed)
    x = source["representations"][indices]
    labels = source[args.label][indices]
    if args.method == "pca":
        from sklearn.decomposition import PCA
        coordinates = PCA(n_components=2, random_state=args.seed).fit_transform(x)
    elif args.method == "tsne":
        from sklearn.manifold import TSNE
        coordinates = TSNE(n_components=2, init="pca", learning_rate="auto",
                           random_state=args.seed).fit_transform(x)
    else:
        try:
            import umap
        except ImportError as error:
            raise RuntimeError("UMAP requires umap-learn; PCA/t-SNE remain available") from error
        coordinates = umap.UMAP(n_components=2, random_state=args.seed).fit_transform(x)
    archive = Path(args.coordinates_out)
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, coordinates=coordinates.astype(np.float32),
                        labels=labels, indices=indices, method=args.method,
                        label=args.label, seed=args.seed)
    if args.figure_out:
        figure = Path(args.figure_out)
        figure.parent.mkdir(parents=True, exist_ok=True)
        _save_plot(coordinates, labels, f"{args.method.upper()} — {args.label}", figure)


if __name__ == "__main__":
    main()
