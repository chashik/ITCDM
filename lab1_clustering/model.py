"""Lab 1. Exploratory clustering of stocks.

Utilities: per-stock feature construction, Hopkins statistic of clustering
tendency, correlation distance, cluster-purity and bootstrap stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

FEATURES = ["ann_return", "ann_vol", "beta_mkt", "idio_vol", "max_drawdown", "skew",
            "beta_rates", "beta_oil", "corr_mkt"]


def stock_features(rets: pd.DataFrame, mkt: pd.Series, d_yield: pd.Series,
                   oil: pd.Series) -> pd.DataFrame:
    """Per-stock descriptors computed from daily log-returns.

    beta_rates : sensitivity to daily changes of the 10-year yield (p.p.);
    beta_oil   : sensitivity to daily WTI returns.  Both are estimated jointly
                 with the market beta in one multiple regression, so they
                 measure exposure *beyond* the market factor.
    """
    out = {}
    Z = pd.concat([mkt.rename("mkt"), d_yield.rename("dy"), oil.rename("oil")], axis=1)
    for t in rets.columns:
        r = rets[t]
        df = pd.concat([r.rename("r"), Z], axis=1).dropna()
        A = np.c_[np.ones(len(df)), df[["mkt", "dy", "oil"]].to_numpy()]
        coef, *_ = np.linalg.lstsq(A, df["r"].to_numpy(), rcond=None)
        resid = df["r"].to_numpy() - A @ coef
        wealth = np.exp(r.cumsum())
        out[t] = {"ann_return": 252 * r.mean(), "ann_vol": np.sqrt(252) * r.std(),
                  "beta_mkt": coef[1], "idio_vol": np.sqrt(252) * resid.std(),
                  "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
                  "skew": float(r.skew()), "beta_rates": coef[2], "beta_oil": coef[3],
                  "corr_mkt": float(np.corrcoef(df["r"], df["mkt"])[0, 1])}
    return pd.DataFrame(out).T[FEATURES]


def hopkins(X: np.ndarray, m: int | None = None, seed: int = 0) -> float:
    """Hopkins statistic H (Han, Kamber & Pei, Sect. 10.6.1 convention).

    H = sum(y) / (sum(x) + sum(y)), where x are nearest-neighbour distances
    of m uniform points (in the data's bounding box) to the data, and y are
    nearest-neighbour distances of m sampled data points to the rest of the
    data.  H ~ 0.5 for uniformly distributed data; H -> 0 for highly
    clustered data (with this orientation of the ratio).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    m = m or max(10, n // 5)
    lo, hi = X.min(0), X.max(0)
    U = rng.uniform(lo, hi, size=(m, d))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    x = nn.kneighbors(U, n_neighbors=1)[0][:, 0]
    idx = rng.choice(n, m, replace=False)
    y = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    return float(y.sum() / (x.sum() + y.sum()))


def corr_distance(rets: pd.DataFrame) -> pd.DataFrame:
    """Mantegna's metric d_ij = sqrt(2 (1 - rho_ij))."""
    rho = rets.corr().clip(-1, 1)
    return np.sqrt(2 * (1 - rho))


def purity(labels_true, labels_pred) -> float:
    ct = pd.crosstab(labels_pred, labels_true)
    return float(ct.max(axis=1).sum() / ct.to_numpy().sum())


def bootstrap_stability(X: np.ndarray, k: int, n_boot: int = 100, seed: int = 0) -> np.ndarray:
    """ARI between the full-sample k-means partition and partitions fitted on
    bootstrap resamples (objects resampled with replacement, then all
    objects assigned to the nearest bootstrap centroid)."""
    rng = np.random.default_rng(seed)
    base = KMeans(k, n_init=20, random_state=seed).fit(X).labels_
    aris = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        km = KMeans(k, n_init=10, random_state=b).fit(X[idx])
        aris.append(adjusted_rand_score(base, km.predict(X)))
    return np.array(aris)
