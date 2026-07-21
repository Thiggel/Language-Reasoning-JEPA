import numpy as np
import pytest

from textjepa.analysis.representations import (
    cluster_alignment,
    effective_rank,
    fit_categorical_probe,
    fit_numeric_probe,
    linear_cka,
)


def test_frozen_numeric_and_categorical_probes_recover_linear_factors():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(200, 8))
    test = rng.normal(size=(80, 8))
    numeric_train, numeric_test = train[:, 0] * 2, test[:, 0] * 2
    class_train, class_test = train[:, 1] > 0, test[:, 1] > 0
    assert fit_numeric_probe(train, numeric_train, test, numeric_test).test_metrics["r2"] > .98
    assert fit_categorical_probe(train, class_train, test, class_test).test_metrics["accuracy"] > .9


def test_geometry_diagnostics_have_known_endpoints():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 6))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert 1.0 <= effective_rank(x) <= 6.0
    labels = (x[:, 0] > 0).astype(int)
    metrics = cluster_alignment(x, labels)
    assert set(metrics) >= {"adjusted_rand", "normalized_mutual_information"}


def test_probe_rejects_tiny_or_constant_inputs():
    with pytest.raises(ValueError, match="ten"):
        fit_numeric_probe(np.ones((5, 2)), np.ones(5), np.ones((5, 2)), np.ones(5))
    with pytest.raises(ValueError, match="constant"):
        fit_categorical_probe(
            np.ones((20, 2)), np.zeros(20), np.ones((10, 2)), np.zeros(10)
        )
