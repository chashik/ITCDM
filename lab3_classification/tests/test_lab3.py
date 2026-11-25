"""Lab 3 tests: perceptron rule, logistic neuron, thresholds, data preparation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from lab3_classification.model import (LogisticNeuron, Perceptron, best_threshold,
                                       expected_cost, prepare, slog)


def separable(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X @ np.array([2.0, -1.0]) + 0.3 > 0).astype(int)
    keep = np.abs(X @ np.array([2.0, -1.0]) + 0.3) > 0.2  # margin
    return X[keep], y[keep]


def test_perceptron_converges_on_separable_data():
    X, y = separable()
    p = Perceptron(epochs=200).fit(X, y)
    assert p.history_[-1][0] == 0            # an epoch without mistakes (Novikoff)
    assert (p.predict(X) == y).all()


def test_pocket_never_worse_than_last_iterate():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 3)); y = (X[:, 0] + rng.normal(0, 1, 500) > 0).astype(int)
    last = Perceptron(epochs=15, seed=1).fit(X, y)
    pocket = Perceptron(epochs=15, seed=1, pocket=True).fit(X, y)
    err = lambda m: np.mean(m.predict(X) != y)
    assert err(pocket) <= err(last) + 1e-12


def test_logistic_neuron_matches_logistic_regression():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, 4)); w = np.array([1.0, -2.0, 0.5, 0.0])
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-(X @ w - 0.5)))).astype(int)
    ln = LogisticNeuron(lr=0.5, epochs=80, l2=0.0).fit(X, y)
    sk = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    assert np.allclose(ln.w_, sk.coef_.ravel(), atol=0.08)
    assert ln.b_ == pytest.approx(sk.intercept_[0], abs=0.08)


def test_cost_optimal_threshold_close_to_bayes_rule():
    # calibrated scores: P(y=1 | s) = s  ->  optimal threshold c_fp / (c_fp + c_fn)
    rng = np.random.default_rng(5)
    s = rng.uniform(size=200_000); y = (rng.uniform(size=s.size) < s).astype(int)
    thr, _, _ = best_threshold(y, s, c_fn=5, c_fp=1, grid=np.linspace(0.01, 0.99, 99))
    assert thr == pytest.approx(1 / 6, abs=0.02)


def test_expected_cost_counts_errors():
    y = np.array([1, 1, 0, 0]); s = np.array([0.9, 0.1, 0.8, 0.2])
    # one FN (cost 5) and one FP (cost 1) over 4 clients
    assert expected_cost(y, s, 0.5, 5, 1) == pytest.approx(6 / 4)


def test_prepare_maps_undocumented_codes_and_keeps_rows():
    df = pd.DataFrame({"LIMIT_BAL": [1e4, 2e4], "SEX": [1, 2], "EDUCATION": [0, 6], "MARRIAGE": [0, 1],
                       "AGE": [30, 40], **{f"PAY_{i}": [-2, 3] for i in range(1, 7)},
                       **{f"BILL_AMT{i}": [-50.0, 1000.0] for i in range(1, 7)},
                       **{f"PAY_AMT{i}": [0.0, 500.0] for i in range(1, 7)}, "default": [0, 1]})
    X, y = prepare(df)
    assert len(X) == 2 and y.tolist() == [0, 1]
    assert X[["edu_graduate", "edu_university", "edu_high_school"]].to_numpy().sum() == 0  # 'other'
    assert X.loc[1, "n_delinquent"] == 6 and X.loc[0, "bill_amt1"] < 0
    assert not X.isna().any().any()


def test_slog_is_odd_and_monotone():
    x = np.linspace(-1e5, 1e5, 101)
    assert np.allclose(slog(-x), -slog(x)) and np.all(np.diff(slog(x)) > 0)
