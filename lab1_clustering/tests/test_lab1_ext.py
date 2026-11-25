"""Lab 1 (iteration 2) tests: BIC choice of a Gaussian mixture, removal of the market factor."""
import numpy as np
from sklearn.mixture import GaussianMixture


def test_bic_recovers_the_number_of_components():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.5, size=(150, 2)) for c in ((0, 0), (4, 0), (0, 4))])
    bic = {k: GaussianMixture(k, covariance_type="tied", n_init=5, random_state=0, reg_covar=1e-2).fit(X).bic(X)
           for k in range(1, 7)}
    assert min(bic, key=bic.get) == 3


def test_market_residuals_remove_the_common_factor():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, 2000)
    beta = rng.uniform(0.5, 1.5, 20)
    R = np.outer(m, beta) + rng.normal(0, 0.005, (2000, 20))
    raw = np.corrcoef(R.T)[np.triu_indices(20, 1)].mean()
    b = np.array([np.polyfit(m, R[:, j], 1)[0] for j in range(20)])
    res = R - np.outer(m, b)
    resid = np.corrcoef(res.T)[np.triu_indices(20, 1)].mean()
    assert raw > 0.7 and abs(resid) < 0.05
