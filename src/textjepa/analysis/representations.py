"""Leakage-resistant quantitative analysis of frozen representations.

Every transform is fit on training representations only.  Visual projections
are intentionally absent: PCA/UMAP plots may illustrate a result, but the
paper's evidence comes from held-out probes and preregistered geometry metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    mean_absolute_error,
    normalized_mutual_info_score,
    r2_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RIDGE_STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LOGISTIC_STRENGTHS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or len(value) < 2:
        raise ValueError("representations must be a two-dimensional sample matrix")
    if not np.isfinite(value).all():
        raise ValueError("representations contain non-finite values")
    return value


def _validation_indices(
    n: int, seed: int, groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n < 10:
        raise ValueError("at least ten training samples are required")
    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(n)
        cut = max(1, int(round(0.8 * n)))
        return order[:cut], order[cut:]
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("group count does not match samples")
    unique = rng.permutation(np.unique(groups))
    if len(unique) < 2:
        raise ValueError("grouped validation requires at least two problems")
    cut = max(1, min(len(unique) - 1, int(round(0.8 * len(unique)))))
    fit_groups = set(unique[:cut].tolist())
    fit = np.asarray([i for i, group in enumerate(groups) if group in fit_groups])
    val = np.asarray([i for i, group in enumerate(groups) if group not in fit_groups])
    return fit, val


@dataclass(frozen=True)
class ProbeResult:
    hyperparameter: float
    validation_score: float
    test_metrics: dict[str, float]


def fit_numeric_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int = 0,
    train_groups: np.ndarray | None = None,
) -> ProbeResult:
    """Select ridge strength on an inner training split, then test once."""
    train_x, test_x = _matrix(train_x), _matrix(test_x)
    train_y = np.asarray(train_y, dtype=np.float64)
    test_y = np.asarray(test_y, dtype=np.float64)
    fit_idx, val_idx = _validation_indices(len(train_x), seed, train_groups)
    candidates = []
    for alpha in RIDGE_STRENGTHS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train_x[fit_idx], train_y[fit_idx])
        pred = model.predict(train_x[val_idx])
        candidates.append((mean_absolute_error(train_y[val_idx], pred), alpha))
    val_error, alpha = min(candidates, key=lambda item: (item[0], item[1]))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return ProbeResult(alpha, -float(val_error), {
        "mae": float(mean_absolute_error(test_y, pred)),
        "r2": float(r2_score(test_y, pred)),
    })


def fit_categorical_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    seed: int = 0,
    train_groups: np.ndarray | None = None,
) -> ProbeResult:
    """Select linear-logistic regularization without looking at test labels."""
    train_x, test_x = _matrix(train_x), _matrix(test_x)
    train_y, test_y = np.asarray(train_y), np.asarray(test_y)
    if np.unique(train_y).size < 2:
        raise ValueError("categorical target is constant in training data")
    fit_idx, val_idx = _validation_indices(len(train_x), seed, train_groups)
    candidates = []
    for strength in LOGISTIC_STRENGTHS:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=strength, max_iter=2000, random_state=seed),
        )
        model.fit(train_x[fit_idx], train_y[fit_idx])
        pred = model.predict(train_x[val_idx])
        score = balanced_accuracy_score(train_y[val_idx], pred)
        candidates.append((-score, strength))
    neg_score, strength = min(candidates, key=lambda item: (item[0], item[1]))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=strength, max_iter=2000, random_state=seed),
    )
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return ProbeResult(strength, -float(neg_score), {
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
    })


def effective_rank(x: np.ndarray) -> float:
    """Entropy effective rank of centered features (collapse diagnostic)."""
    x = _matrix(x) - np.mean(x, axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    mass = np.square(singular)
    if mass.sum() == 0:
        return 0.0
    probability = mass / mass.sum()
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered-kernel alignment for aligned examples."""
    x, y = _matrix(x), _matrix(y)
    if len(x) != len(y):
        raise ValueError("CKA requires aligned sample counts")
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(
        y.T @ y, ord="fro"
    )
    return float(cross / denom) if denom else 0.0


def cluster_alignment(x: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """Unsupervised cluster agreement with a predeclared discrete factor."""
    x, labels = _matrix(x), np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2 or len(classes) >= len(labels):
        raise ValueError("cluster factor needs between two and n-1 classes")
    standardized = StandardScaler().fit_transform(x)
    predicted = KMeans(
        n_clusters=len(classes), n_init=20, random_state=seed
    ).fit_predict(standardized)
    result = {
        "adjusted_rand": float(adjusted_rand_score(labels, predicted)),
        "normalized_mutual_information": float(
            normalized_mutual_info_score(labels, predicted)
        ),
    }
    if len(classes) < len(labels) - 1:
        result["silhouette"] = float(silhouette_score(standardized, predicted))
    return result
