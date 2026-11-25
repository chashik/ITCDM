"""Lab 2 tests: polynomial pipeline, clipping, PCA, leakage-free CV."""
import numpy as np
import pytest

from lab2_regression.model import ClipToTrainingRange, cv_score, make_model, metrics, n_terms


@pytest.fixture
def poly_data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(600, 3))
    y = 1.0 + 2 * X[:, 0] - X[:, 1] ** 2 + 0.5 * X[:, 0] * X[:, 2] + rng.normal(0, 0.1, 600)
    return X, y


def test_number_of_monomials():
    # C(p+d, d) - 1 monomials without the intercept
    assert n_terms(7, 1) == 7 and n_terms(7, 2) == 35 and n_terms(7, 3) == 119


def test_degree_two_recovers_quadratic_surface(poly_data):
    X, y = poly_data
    lin = make_model(1).fit(X[:400], y[:400])
    quad = make_model(2).fit(X[:400], y[:400])
    r_lin = metrics(y[400:], lin.predict(X[400:]))["RMSE"]
    r_quad = metrics(y[400:], quad.predict(X[400:]))["RMSE"]
    assert r_quad < 0.15 < r_lin


def test_cv_prefers_true_degree(poly_data):
    X, y = poly_data
    scores = {d: cv_score(lambda: make_model(d), X, y, gap=0)[1].mean() for d in (1, 2, 3)}
    assert min(scores, key=scores.get) in (2, 3)
    assert scores[2] < 0.5 * scores[1]


def test_clipping_bounds_extrapolation():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, size=(200, 1))
    y = X[:, 0] ** 3
    m = make_model(4).fit(X, y)
    mc = make_model(4, clip=True).fit(X, y)
    far = np.array([[25.0]])
    edge = mc.predict(np.array([[X.max()]]))[0]
    assert mc.predict(far)[0] == pytest.approx(edge)  # constant beyond the range
    assert abs(m.predict(far)[0]) > 100               # raw polynomial explodes


def test_clip_transformer_learns_range_from_fit_only():
    c = ClipToTrainingRange().fit(np.array([[0.0], [1.0]]))
    assert c.transform(np.array([[5.0], [-3.0]])).ravel().tolist() == [1.0, 0.0]


def test_scaler_is_fitted_inside_pipeline(poly_data):
    X, y = poly_data
    m = make_model(2, n_pca=2).fit(X[:300], y[:300])
    assert np.allclose(m.named_steps["scale"].mean_, X[:300].mean(axis=0))


def test_adaptive_conformal_recovers_coverage_after_a_shift():
    from lab2_regression.run_ext import adaptive_conformal
    rng = np.random.default_rng(7)
    cal = np.abs(rng.normal(size=500))
    y = np.r_[rng.normal(size=1500), 2.0 * rng.normal(size=3000)]     # volatility doubles at t = 1500
    pred = np.zeros_like(y)
    q = np.quantile(cal, 0.9)
    widths, _ = adaptive_conformal(pred, y, cal, h=5, alpha=0.1, gamma=0.01, window=300)
    late = slice(2500, None)
    split_cov = np.mean(np.abs(y[late]) <= q)
    adapt_cov = np.mean(np.abs(y[late]) <= widths[late])
    assert split_cov < 0.75 and abs(adapt_cov - 0.9) < 0.04
