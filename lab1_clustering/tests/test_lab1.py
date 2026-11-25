"""Lab 1 tests: clustering tendency, distances, features, stability."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from lab1_clustering.model import bootstrap_stability, corr_distance, hopkins, purity, stock_features


def test_hopkins_uniform_vs_clustered():
    rng = np.random.default_rng(0)
    U = rng.uniform(size=(400, 3))
    B, _ = make_blobs(400, n_features=3, centers=5, cluster_std=0.05, center_box=(0, 1), random_state=0)
    hu = np.mean([hopkins(U, seed=s) for s in range(20)])
    hb = np.mean([hopkins(B, seed=s) for s in range(20)])
    assert 0.42 < hu < 0.58     # ~0.5 for spatially random data
    assert hb < 0.2             # -> 0 for clustered data (Han et al. orientation)


def test_corr_distance_is_a_metric_on_examples():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    R = pd.DataFrame({"a": x, "b": x, "c": -x, "d": rng.normal(size=500)})
    D = corr_distance(R)
    assert D.loc["a", "b"] == pytest.approx(0, abs=1e-6)
    assert D.loc["a", "c"] == pytest.approx(2, abs=1e-6)
    assert D.loc["a", "d"] == pytest.approx(np.sqrt(2), abs=0.15)
    assert np.allclose(D, D.T)


def test_purity():
    assert purity([0, 0, 1, 1], [5, 5, 7, 7]) == 1.0
    assert purity([0, 1, 0, 1], [5, 5, 5, 5]) == 0.5


def test_bootstrap_stability_high_for_true_k():
    X, _ = make_blobs(300, centers=4, cluster_std=0.5, random_state=2)
    s4 = bootstrap_stability(X, 4, n_boot=20)
    s7 = bootstrap_stability(X, 7, n_boot=20)
    assert s4.mean() > 0.95 and s4.mean() > s7.mean()


def test_stock_features_recover_betas():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2020-01-01", periods=1500)
    mkt = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    dy = pd.Series(rng.normal(0, 0.05, len(idx)), index=idx)
    oil = pd.Series(rng.normal(0, 0.02, len(idx)), index=idx)
    r = 1.5 * mkt - 0.05 * dy + 0.3 * oil + rng.normal(0, 0.005, len(idx))
    F = stock_features(pd.DataFrame({"X": r}), mkt, dy, oil)
    assert F.loc["X", "beta_mkt"] == pytest.approx(1.5, abs=0.05)
    assert F.loc["X", "beta_rates"] == pytest.approx(-0.05, abs=0.01)
    assert F.loc["X", "beta_oil"] == pytest.approx(0.3, abs=0.03)
    assert -1 <= F.loc["X", "max_drawdown"] <= 0
