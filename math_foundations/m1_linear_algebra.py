"""Part 1, chapter 2 (mathematical foundations), section M1. Linear algebra of least squares, ridge and PCA on the data of Lab 2 (volatility regression).

1. Conditioning: kappa(X^T X) = kappa(X)^2, so solving the normal equations loses about twice as many digits as a QR
   (or SVD) solution of the least-squares problem (Boyd and Vandenberghe; Goodfellow et al., ch. 4).  Polynomial
   features of degree 3 of the seven Lab 2 predictors, raw and standardized.
2. Ridge regression through the SVD X = U D V^T: beta = V diag(d / (d^2 + lambda)) U^T y; the shrinkage factors
   d^2 / (d^2 + lambda) and the effective degrees of freedom df = sum d^2 / (d^2 + lambda); compared with scikit-learn.
3. PCA and the Eckart-Young theorem: the best rank-k approximation of the centred data in the Frobenius norm is the
   truncated SVD, and its squared error is the sum of the discarded squared singular values.
Outputs: paper/generated/math1_numbers.tex.  Run (from code/):  python -m math_foundations.m1_linear_algebra
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from dmfin.reporting import save_numbers, sci


def lab2_train():
    from dmfin.data import load_market_panel
    from dmfin.features import add_targets, market_features
    from dmfin.splits import chrono_split
    from lab2_regression.model import FEATURES, HORIZON, TARGET, design
    d = design(add_targets(market_features(load_market_panel())))
    sp = chrono_split(d.index, horizon=HORIZON)
    tr, te = d.loc[sp.train], d.loc[sp.test]
    return tr[FEATURES].to_numpy(), tr[TARGET].to_numpy(), te[FEATURES].to_numpy(), te[TARGET].to_numpy()


def normal_vs_qr(X: np.ndarray, y: np.ndarray, Xte: np.ndarray, yte: np.ndarray) -> dict:
    k = np.linalg.cond(X)
    b_qr = np.linalg.lstsq(X, y, rcond=None)[0]              # SVD-based, backward stable
    b_ne = np.linalg.solve(X.T @ X, X.T @ y)                 # normal equations
    rmse = lambda b: float(np.sqrt(np.mean((Xte @ b - yte) ** 2)))
    resid = lambda b: float(np.linalg.norm(X.T @ (X @ b - y)) / np.linalg.norm(X.T @ y))   # optimality of the solution
    return {"cond": sci(k), "cond_gram": sci(np.linalg.cond(X.T @ X)), "digits": float(np.log10(k)),
            "coef_reldiff": float(np.linalg.norm(b_ne - b_qr) / np.linalg.norm(b_qr)),
            "opt_qr": resid(b_qr), "opt_ne": resid(b_ne), "rmse_qr": rmse(b_qr), "rmse_ne": rmse(b_ne)}


def main():
    num = {}
    Xtr, ytr, Xte, yte = lab2_train()
    num["n_train"], num["p"] = Xtr.shape
    # 1. conditioning of raw and standardized cubic features
    for tag, scale in (("raw", False), ("std", True)):
        sc = StandardScaler().fit(Xtr) if scale else None
        f = (lambda Z: sc.transform(Z)) if scale else (lambda Z: Z)
        poly = PolynomialFeatures(3, include_bias=True).fit(f(Xtr))
        A, Ate = poly.transform(f(Xtr)), poly.transform(f(Xte))
        for k, v in normal_vs_qr(A, ytr, Ate, yte).items():
            num[f"{tag}_{k}"] = v
    num["n_terms_d3"] = A.shape[1]
    # 2. ridge through the SVD (standardized quadratic features, centred target)
    Z = PolynomialFeatures(2, include_bias=False).fit_transform(StandardScaler().fit_transform(Xtr))
    Z = (Z - Z.mean(0)) / Z.std(0)
    yc = ytr - ytr.mean()
    U, dsv, Vt = np.linalg.svd(Z, full_matrices=False)
    lam = 100.0
    beta = Vt.T @ ((dsv / (dsv ** 2 + lam)) * (U.T @ yc))
    sk = Ridge(alpha=lam, fit_intercept=False).fit(Z, yc).coef_
    shrink = dsv ** 2 / (dsv ** 2 + lam)
    num.update({"ridge_lambda": lam, "ridge_terms": Z.shape[1], "ridge_svd_vs_sklearn": float(np.abs(beta - sk).max()),
                "ridge_df": float(shrink.sum()), "ridge_shrink_min": float(shrink.min()), "ridge_shrink_max": float(shrink.max()),
                "ridge_cond": float(dsv[0] / dsv[-1])})
    # 3. PCA and Eckart-Young on the seven standardized predictors
    S = StandardScaler().fit_transform(Xtr)
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    for k in (1, 2, 3):
        approx = (U[:, :k] * s[:k]) @ Vt[:k]
        err = float(np.linalg.norm(S - approx, "fro") ** 2)
        num[f"ey_err_k{k}"] = err
        num[f"ey_tail_k{k}"] = float((s[k:] ** 2).sum())
        num[f"ey_explained_k{k}"] = 100 * float((s[:k] ** 2).sum() / (s ** 2).sum())
    num["ey_max_gap"] = max(abs(num[f"ey_err_k{k}"] - num[f"ey_tail_k{k}"]) / num[f"ey_tail_k{k}"] for k in (1, 2, 3))
    save_numbers("math1", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
