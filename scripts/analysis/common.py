"""Shared numerics for the representation-analysis battery.

Pure torch/CPU, no matplotlib (import safe inside ``.venv``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, row-wise. ``a``/``b``: [N, D]."""
    a = a.double()
    b = b.double()
    num = (a * b).sum(-1)
    den = a.norm(dim=-1).clamp_min(1e-12) * b.norm(dim=-1).clamp_min(1e-12)
    return 1.0 - num / den


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.double() - b.double()).norm(dim=-1)


def normalized_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """L2 after per-vector RMS normalisation (scale-free across encoders)."""
    def rms(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (rms(a.double()) - rms(b.double())).norm(dim=-1)


DISTANCES = {
    "cosine": cosine_distance,
    "l2": l2_distance,
    "l2_normalized": normalized_l2,
}


# --------------------------------------------------------------------------- #
# AUC (tie-corrected Mann-Whitney)
# --------------------------------------------------------------------------- #
def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """P(score of a positive > score of a negative), ties counted as 1/2."""
    scores = scores.double().flatten()
    labels = labels.double().flatten()
    pos, neg = labels > 0.5, labels <= 0.5
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.double)
    vals, inverse, counts = scores.unique(return_inverse=True, return_counts=True)
    sums = torch.zeros(len(vals), dtype=torch.double).index_add_(0, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def bootstrap_ci(
    values: torch.Tensor, n_boot: int = 1000, seed: int = 0, q: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values``."""
    values = values.double().flatten()
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = values[idx].mean(1)
    lo = float(means.quantile((1 - q) / 2))
    hi = float(means.quantile(1 - (1 - q) / 2))
    return (lo, hi)


@dataclass
class SeparationResult:
    """Same-consequence (``same``) vs different-consequence (``diff``) distances."""

    n_same: int
    n_diff: int
    same_mean: float
    diff_mean: float
    same_median: float
    diff_median: float
    ratio: float          # diff_mean / same_mean; >1 means the claim holds
    auc: float            # AUC of "different-consequence" from distance alone
    cohens_d: float
    same_ci: tuple[float, float] = (float("nan"), float("nan"))
    diff_ci: tuple[float, float] = (float("nan"), float("nan"))

    def to_dict(self) -> dict:
        return {
            "n_same": self.n_same, "n_diff": self.n_diff,
            "same_mean": self.same_mean, "diff_mean": self.diff_mean,
            "same_median": self.same_median, "diff_median": self.diff_median,
            "separation_ratio": self.ratio, "auc_diff_vs_same": self.auc,
            "cohens_d": self.cohens_d,
            "same_ci95": list(self.same_ci), "diff_ci95": list(self.diff_ci),
        }


def separation(
    same: torch.Tensor, diff: torch.Tensor, seed: int = 0
) -> SeparationResult:
    """Summarise two distance populations.

    ``same``  distances between representations that SHOULD collapse
    ``diff``  distances between representations that SHOULD separate

    ``auc`` is the probability that a random different-consequence pair is
    farther apart than a random same-consequence pair; .5 = no signal.
    """
    same = same.double().flatten()
    diff = diff.double().flatten()
    if len(same) == 0 or len(diff) == 0:
        return SeparationResult(len(same), len(diff), *([float("nan")] * 6))
    scores = torch.cat([same, diff])
    labels = torch.cat([torch.zeros(len(same)), torch.ones(len(diff))])
    sm, dm = float(same.mean()), float(diff.mean())
    pooled = math.sqrt(
        (float(same.var(unbiased=True)) + float(diff.var(unbiased=True))) / 2
    ) if len(same) > 1 and len(diff) > 1 else float("nan")
    return SeparationResult(
        n_same=len(same), n_diff=len(diff),
        same_mean=sm, diff_mean=dm,
        same_median=float(same.median()), diff_median=float(diff.median()),
        ratio=(dm / sm) if sm > 1e-12 else float("nan"),
        auc=auc(scores, labels),
        cohens_d=((dm - sm) / pooled) if pooled and pooled > 1e-12 else float("nan"),
        same_ci=bootstrap_ci(same, seed=seed),
        diff_ci=bootstrap_ci(diff, seed=seed + 1),
    )


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def mlp_probe(d_in: int, hidden: int = 128, d_out: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, d_out),
    )


def standardize(Xtr: torch.Tensor, *others: torch.Tensor):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-5)
    return ((Xtr - mu) / sd, *[(o - mu) / sd for o in others])


def _fit(model, Xtr, ytr, Xva, yva, lossf, epochs, bs, lr, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = math.inf, None, 0
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xtr[j]), ytr[j]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(lossf(model(Xva), yva))
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return best, model(Xva)


def train_binary_probe(
    kind: str, Xtr, ytr, Xva, yva, seed: int = 0,
    epochs: int = 40, bs: int = 4096, lr: float = 3e-3, hidden: int = 128,
):
    """``kind`` in {"linear", "mlp"}; returns val scores [N]."""
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = nn.Linear(d, 1) if kind == "linear" else mlp_probe(d, hidden)
    lossf = nn.BCEWithLogitsLoss()
    _, scores = _fit(
        model, Xtr, ytr.unsqueeze(-1), Xva, yva.unsqueeze(-1),
        lossf, epochs, bs, lr,
    )
    return scores.squeeze(-1)


def train_regression_probe(
    kind: str, Xtr, ytr, Xva, yva, seed: int = 0,
    epochs: int = 60, bs: int = 4096, lr: float = 3e-3, hidden: int = 128,
):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = nn.Linear(d, 1) if kind == "linear" else mlp_probe(d, hidden)
    lossf = nn.MSELoss()
    _, pred = _fit(
        model, Xtr, ytr.unsqueeze(-1), Xva, yva.unsqueeze(-1),
        lossf, epochs, bs, lr,
    )
    return pred.squeeze(-1)


def train_multiclass_probe(
    kind: str, Xtr, ytr, Xva, yva, n_class: int, seed: int = 0,
    epochs: int = 60, bs: int = 4096, lr: float = 3e-3, hidden: int = 128,
):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = nn.Linear(d, n_class) if kind == "linear" else mlp_probe(d, hidden, n_class)
    lossf = nn.CrossEntropyLoss()
    _, logits = _fit(model, Xtr, ytr, Xva, yva, lossf, epochs, bs, lr)
    return logits


# --------------------------------------------------------------------------- #
# metric summaries
# --------------------------------------------------------------------------- #
def binary_summary(scores, labels, depth=None) -> dict:
    labels = labels.double()
    out = {
        "n": int(len(labels)),
        "auc": auc(scores, labels),
        "acc": float(((scores > 0).double() == labels).double().mean()),
        "pos_rate": float(labels.mean()),
        "majority_acc": float(max(labels.mean(), 1 - labels.mean())),
    }
    if depth is not None:
        out["by_depth"] = {}
        for d in sorted(set(int(x) for x in depth.tolist())):
            m = depth == d
            if int(m.sum()) < 50:
                continue
            out["by_depth"][int(d)] = {
                "n": int(m.sum()),
                "auc": auc(scores[m], labels[m]),
                "acc": float(
                    ((scores[m] > 0).double() == labels[m]).double().mean()
                ),
            }
    return out


def regression_summary(pred, target, depth=None) -> dict:
    pred = pred.double()
    target = target.double()
    resid = pred - target
    var = float(target.var(unbiased=False))
    out = {
        "n": int(len(target)),
        "mae": float(resid.abs().mean()),
        "rmse": float((resid ** 2).mean().sqrt()),
        "r2": float(1 - (resid ** 2).mean() / var) if var > 1e-12 else float("nan"),
        "target_std": float(var ** 0.5),
    }
    if depth is not None:
        out["by_depth"] = {}
        for d in sorted(set(int(x) for x in depth.tolist())):
            m = depth == d
            if int(m.sum()) < 50:
                continue
            r = resid[m]
            v = float(target[m].var(unbiased=False))
            out["by_depth"][int(d)] = {
                "n": int(m.sum()), "mae": float(r.abs().mean()),
                "r2": float(1 - (r ** 2).mean() / v) if v > 1e-12 else float("nan"),
            }
    return out


def multiclass_summary(logits, labels, depth=None) -> dict:
    pred = logits.argmax(-1)
    counts = torch.bincount(labels, minlength=logits.shape[1]).double()
    out = {
        "n": int(len(labels)),
        "acc": float((pred == labels).double().mean()),
        "majority_acc": float(counts.max() / counts.sum()),
        "n_class": int(logits.shape[1]),
    }
    if depth is not None:
        out["by_depth"] = {}
        for d in sorted(set(int(x) for x in depth.tolist())):
            m = depth == d
            if int(m.sum()) < 50:
                continue
            out["by_depth"][int(d)] = {
                "n": int(m.sum()),
                "acc": float((pred[m] == labels[m]).double().mean()),
            }
    return out
