"""Linear probes over frozen latents.

``ridge_probe_accuracy`` is a fast closed-form probe used during training
evals; ``logistic_probe_accuracy`` (sklearn) is the thorough offline probe.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def ridge_probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    l2: float = 1e-1,
    train_frac: float = 0.8,
    max_n: int = 20000,
    seed: int = 0,
) -> float:
    """Closed-form ridge regression to one-hot targets; returns test accuracy."""
    x = features.float()
    y = labels.long()
    if x.shape[0] > max_n:
        idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
        x, y = x[idx[:max_n].to(x.device)], y[idx[:max_n].to(y.device)]
    n = x.shape[0]
    n_train = int(n * train_frac)
    if n_train < n_classes or n - n_train < 1:
        return float("nan")
    x = (x - x[:n_train].mean(0)) / (x[:n_train].std(0) + 1e-6)
    x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
    xt, yt = x[:n_train], y[:n_train]
    onehot = torch.zeros(n_train, n_classes, device=x.device)
    onehot[torch.arange(n_train), yt] = 1.0
    a = xt.T @ xt + l2 * torch.eye(x.shape[1], device=x.device)
    w = torch.linalg.solve(a, xt.T @ onehot)
    pred = (x[n_train:] @ w).argmax(dim=1)
    return (pred == y[n_train:]).float().mean().item()


def logistic_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.8,
    max_iter: int = 2000,
    c: float = 1.0,
    seed: int = 0,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(labels))
    n_train = int(len(labels) * train_frac)
    tr, te = idx[:n_train], idx[n_train:]
    if len(np.unique(labels[tr])) < 2:
        return float("nan")
    scaler = StandardScaler().fit(features[tr])
    clf = LogisticRegression(max_iter=max_iter, C=c)
    clf.fit(scaler.transform(features[tr]), labels[tr])
    return float(clf.score(scaler.transform(features[te]), labels[te]))


def majority_baseline(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())
