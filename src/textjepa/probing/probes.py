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


def ridge_regression_r2(
    features: np.ndarray,
    targets: np.ndarray,
    train_frac: float = 0.8,
    alpha: float = 1.0,
    seed: int = 0,
) -> float:
    """Linear ridge regression; returns test R^2 (continuous targets,
    e.g. cos/sin of the mod-p value for circular-coding probes)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(targets))
    n_train = int(len(targets) * train_frac)
    tr, te = idx[:n_train], idx[n_train:]
    scaler = StandardScaler().fit(features[tr])
    reg = Ridge(alpha=alpha).fit(scaler.transform(features[tr]), targets[tr])
    return float(reg.score(scaler.transform(features[te]), targets[te]))


def mlp_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    hidden: int = 256,
    epochs: int = 200,
    train_frac: float = 0.8,
    seed: int = 0,
) -> float:
    """2-layer MLP probe (torch, full-batch Adam). The gap to the linear
    probe measures information that is present but not linearized."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(labels))
    n_train = int(len(labels) * train_frac)
    tr, te = idx[:n_train], idx[n_train:]
    x = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    x = (x - mu) / sd
    n_classes = int(y.max().item()) + 1
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden),
        torch.nn.GELU(),
        torch.nn.Linear(hidden, n_classes),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x[tr]), y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (net(x[te]).argmax(1) == y[te]).float().mean().item()
    return float(acc)
